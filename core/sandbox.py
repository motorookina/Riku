from __future__ import annotations

import io
import logging
import tarfile
import threading
import uuid
from typing import Any

import docker
from docker.models.containers import Container

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from config import config

logger = logging.getLogger(__name__)

# 不传image时，沙箱就用这个image创建容器。
_DEFAULT_IMAGE = config._DEFAULT_IMAGE
# 容器内默认的工作目录，保证命令在该目录下执行。
_DEFAULT_WORKING_DIR = config._DEFAULT_WORKING_DIR
# 单条命令最大运行时间，超出该时间直接返回超时报错信息（并没有
# 直接杀死该线程，超时后线程还在后台跑）
_DEFAULT_EXECUTE_TIMEOUT = int(config._DEFAULT_EXECUTE_TIMEOUT)
# 输出最大字节数
_DEFAULT_MAX_OUTPUT_BYTES = int(config._DEFAULT_MAX_OUTPUT_BYTES)
# 默认容器名称，防止重复创建
_DEFAULT_CONTAINER_NAME = config._DEFAULT_CONTAINER_NAME

class DockerSandbox(BaseSandbox):
    """Docker 容器沙箱。

    通过 Docker SDK 管理容器的完整生命周期。所有 `execute()`、
    `upload_files()` 和 `download_files()` 调用均在容器内部执行。

    `BaseSandbox` 的子类必须实现 `execute()`、`upload_files()`、
    `download_files()` 以及 `id` 属性——而本类提供了以上全部实现。

    `read()`、`write()`、`edit()`、`ls()`、`delete()`、`grep()` 和
    `glob()`（以及对应的 `a*` 异步版本）无需在子类中重写：
    `BaseSandbox` 已经基于 `execute()` / `upload_files()` /
    `download_files()` 提供了开箱即用的默认实现，本类直接继承即可。
    这些默认实现通过容器内的 `python3` 脚本运行，因此镜像需自带 `python3`。

    参数：
        image: Docker 镜像名称。
        container_name: 容器名称。为 ``None`` 时自动生成。
        volumes: 卷挂载配置，例如
            ``{"/host": {"bind": "/container", "mode": "rw"}}``。
        working_dir: 容器内的工作目录。
        auto_remove: 在 ``close()`` 时是否自动删除容器。
        execute_timeout: ``execute()`` 的默认超时时间（秒）。
        max_output_bytes: 截断输出前的最大输出字节数。
        docker_client_kwargs: 传递给 ``docker.from_env()`` 的额外关键字参数，仅传递docker客户端连接的参数
        env_vars: 传入容器内的环境变量。容器创建后新增/修改的字段会通过
            ``exec_run`` 动态注入到后续每条命令中，无需重建容器。
        recreate_on_env_drift: 环境变量漂移时是否重建容器（默认 False）。
            既有容器的环境变量与当前 ``env_vars`` 不一致时：
            True → 删除旧容器并按新 ``env_vars`` 重建（会丢失容器内未挂载的改动）；
            False → 仅记录警告，靠 exec 动态注入保证命令拿到的环境变量始终最新。
    """

    def __init__(self,
                 image: str = _DEFAULT_IMAGE,
                 container_name: str = _DEFAULT_CONTAINER_NAME,
                 volumes: dict[str, dict[str, str]] | None = None,
                 working_dir: str = _DEFAULT_WORKING_DIR,
                 auto_remove: bool = False,
                 execute_timeout: int = _DEFAULT_EXECUTE_TIMEOUT,
                 max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
                 docker_client_kwargs: dict[str, Any] | None = None,
                 env_vars: dict[str, str] | None = None,
                 recreate_on_env_drift: bool = False, # env_vars 变化时是否重建容器
                 cpu_limit:int | None = None, # CPU核心限制数量
                 memory_limit: str | None = None, # 内存大小限制
                 ):
        # ---------- 1. 保存初始化参数为实例属性 ----------
        self._image = image
        # 若未提供容器名，自动生成唯一名称（前缀 + 随机12位hex）
        self._container_name = container_name or f"sandbox-{uuid.uuid4().hex[:12]}"
        self._volumes = volumes or {}
        self._working_dir = working_dir
        self._auto_remove = auto_remove
        self._default_timeout = execute_timeout
        self._max_output_bytes = max_output_bytes
        self._env_vars=env_vars
        self._recreate_on_env_drift = recreate_on_env_drift
        # 防御性转换：.env 的 CPU_LIMIT 经 os.getenv 读出来是字符串，
        # 参与算术运算（nano_cpus = 核心数*1e9）前必须先转数值；解析失败视为不限制
        if cpu_limit is not None:
            try:
                cpu_limit = float(cpu_limit)
            except (TypeError, ValueError):
                logger.warning("无法解析 cpu_limit=%r，将不限制 CPU", cpu_limit)
                cpu_limit = None
        self._cpu_limit = cpu_limit
        self._memory_limit = memory_limit

        # ---------- 2. 建立与 Docker 守护进程的连接 ----------
        # 从环境变量（如 DOCKER_HOST）或默认套接字读取配置
        # 只负责读取docker守护进程的环境变量，如果使用docker desktop，则无需手动传入
        client_kwargs = docker_client_kwargs or {}
        self._client = docker.from_env(**client_kwargs)

        # ---------- 3. 确保镜像可用（本地不存在则尝试拉取） ----------
        self._check_image()  # 私有方法，具体实现未在此处展示

        # ---------- 4. 创建容器（但尚未启动） ----------
        self._container = self._get_or_create_container()  # 启用或构建容器对象

        # ---------- 5. 在容器内创建工作目录 ----------
        # 执行 mkdir -p，确保工作目录存在
        self.execute(f"mkdir -p {working_dir}")

        # ---------- 6. 记录初始化完成日志 ----------
        logger.info(
            "DockerSandbox initialized: container=%s, image=%s",
            self._container.short_id,
            image,
        )

    def _check_image(self) -> None:
        """检查 Docker 镜像是否存在于本地。

        抛出：
            DockerImageNotFound: 本地没有该镜像。
        """
        try:
            # 调用 Docker SDK 的 images.get() 方法，尝试获取指定名称/标签的镜像对象。
            # 若镜像存在，则该方法会返回 Image 对象（此处未使用返回值，仅用于检查）。
            self._client.images.get(self._image)
        except docker.errors.ImageNotFound:
            # 捕获 Docker SDK 抛出的“镜像未找到”异常。
            # 将其转换为自定义的 DockerImageNotFound 异常，向上层抛出。
            # 使用 "from None" 隐藏原始异常链，避免回溯信息中包含底层 docker 库的内部细节，
            # 使错误栈更清晰，只显示当前应用层定义的异常。
            raise DockerImageNotFound(self._image) from None

    def _effective_env(self) -> dict[str, str] | None:
        """返回实际注入容器的环境变量：剔除值为 None 的键。

        None 值没有意义，直接传入会让 docker 注入字符串 "None" 污染环境；
        统一在这里剔除，保证容器创建（environment=）与命令执行
        （exec_run environment=）两侧看到的环境一致，便于 _env_matches 漂移比对。
        """
        if not self._env_vars:
            return None
        effective = {k: str(v) for k, v in self._env_vars.items() if v is not None}
        return effective or None

    def _env_matches(self, container: Container) -> bool:
        """检查既有容器的环境变量是否与当前 env_vars 一致。

        Docker 容器创建后环境变量不可修改，这里用于检测 env_vars 在容器创建后
        是否被改动。只比对 env_vars 中显式设置（非 None）的键，镜像自带的环境
        变量不参与比对。比对异常时保守返回 True，避免误删/误重建容器。
        """
        expected = self._effective_env()
        if not expected:
            return True
        try:
            current_env: dict[str, str] = {}
            for entry in container.attrs["Config"]["Env"] or []:
                key, _, value = entry.partition("=")
                if key:
                    current_env[key] = value
            return all(current_env.get(k) == v for k, v in expected.items())
        except Exception:
            return True

    def _get_or_create_container(self) -> Container:
        """获取已存在的容器，若不存在则新建。

        环境变量在容器创建后是只读的（Docker 语义）。若既有容器的环境变量与
        当前 env_vars 不一致，说明 env_vars 在创建之后被改过：
          - recreate_on_env_drift=True 时删除旧容器并按新 env_vars 重建；
          - 否则仅记录警告——命令执行层面由 execute() 通过 exec_run(environment=...)
            动态注入最新 env_vars，新增/修改的字段无需重建容器即可立即生效。
        """
        try:
            # 尝试根据名称获取已有容器
            container = self._client.containers.get(self._container_name)
            logger.info(f"Found existing container: {self._container_name}")

            # 环境变量漂移检测：env_vars 有新增/修改但容器已创建
            if not self._env_matches(container):
                if self._recreate_on_env_drift:
                    try:
                        container.remove(force=True)
                        logger.warning(
                            "env_vars 与既有容器不一致，已删除旧容器并按新 env_vars 重建 %s",
                            self._container_name,
                        )
                        return self._create_container()
                    except Exception as e:
                        logger.warning(
                            "重建容器失败（%s），继续复用既有容器；"
                            "命令仍会通过 exec 动态注入最新 env_vars",
                            e,
                        )
                else:
                    logger.warning(
                        "既有容器 %s 的环境变量与当前 env_vars 不一致"
                        "（env_vars 在容器创建后被修改）。已通过 exec_run 动态注入"
                        "最新 env_vars，命令执行即可生效，无需重建容器；"
                        "如需同步容器级环境变量，可设置 recreate_on_env_drift=True。",
                        self._container_name,
                    )

            # 如果容器处于停止状态，重新启动它
            if container.status != "running":
                container.start()
                logger.info(f"Started existing container: {self._container_name}")

            return container
        except docker.errors.NotFound:
            # 容器不存在，创建新的
            logger.info(f"Container {self._container_name} not found. Creating new one.")
            return self._create_container()

    def _create_container(self) -> Container:
        """创建并启动容器。

        容器的启动命令设置为 `tail -f /dev/null`，这是一个永不结束的空操作进程。
        这样做的目的是让容器保持运行状态（而不会执行完任务后立即退出），
        以便后续可以通过 `exec_run` 动态注入实际命令执行。
        """
        # containers.run() 是 Docker SDK 的高阶 API，相当于 docker run 命令
        return self._client.containers.run(
            image=self._image,  # 使用的镜像
            name=self._container_name,  # 容器名称
            command="tail -f /dev/null",  # 保活命令：持续阻塞，使容器保持运行
            working_dir=self._working_dir,  # 默认工作目录
            volumes=self._volumes,  # 挂载卷（宿主机目录 -> 容器目录）
            detach=True,  # 后台运行（不阻塞当前线程）
            stdin_open=True,  # 保持 STDIN 打开（便于交互式调试）
            tty=False,  # 不分配伪终端（避免输出中的控制字符污染日志）
            environment=self._effective_env(), # 向容器内传入环境变量（剔除 None 值）
            # nano_cpus 纳秒级CPU = 核心数*1e9；None 表示不限制
            nano_cpus=int(self._cpu_limit*1e9) if self._cpu_limit is not None else None,
            mem_limit=self._memory_limit,
        )


    @property
    def id(self) -> str:
        """容器短 ID（前 12 位哈希值），便于日志标识和快速引用。"""
        return self._container.short_id

    #工具
    def execute(
            self,
            command: str,
            *,
            timeout: int | None = None,
    ) -> ExecuteResponse:
        """在容器内执行一条 Shell 命令。

        参数：
            command: 要执行的 Shell 命令字符串。
            timeout: 超时时间（秒），为 ``None`` 时使用默认超时。

        返回：
            `ExecuteResponse` 对象，包含合并后的输出、退出码和截断标志。
        """
        # ---------- 1. 输入校验 ----------
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        # ---------- 2. 确定有效超时（传入参数优先，否则用默认值） ----------
        effective_timeout = timeout if timeout is not None else self._default_timeout

        # ---------- 3. 准备线程间通信的共享容器 ----------
        result_holder: dict[str, Any] = {}  # 用于在线程间传递执行结果
        execution_done = threading.Event()  # 事件标志，用于等待子线程完成

        # ---------- 4. 定义子线程的执行函数 ----------
        def run_command() -> None:
            try:
                # exec_run 是 Docker SDK 的低阶 API，在已运行的容器中执行新进程
                # 相当于 docker exec 命令
                exit_code, output = self._container.exec_run(
                    cmd=["sh", "-c", command],  # 通过 Shell 执行命令，支持管道、重定向等
                    stdout=True,  # 捕获标准输出
                    stderr=True,  # 捕获标准错误（与 stdout 合并返回）
                    demux=False,  # 不分离 stdout/stderr，合并为一个字节流
                    workdir=self._working_dir,  # 指定执行目录，覆盖容器的默认工作目录
                    # 动态注入环境变量：容器创建后新增/修改的 env_vars 字段，
                    # 每条命令执行时也一并生效，无需重建容器
                    # （exec 级环境变量与容器级叠加：覆盖同名变量、保留其余容器变量）
                    environment=self._effective_env(),
                )
                # 将结果写入共享字典
                result_holder["exit_code"] = exit_code
                # 解码输出字节流，遇到无法解码的字符用 Unicode 替换符（�）代替
                result_holder["output"] = (
                    output.decode("utf-8", errors="replace") if output else ""
                )
            except Exception as e:
                # 捕获所有异常（如容器被停止、命令不存在等），避免子线程崩溃
                result_holder["exit_code"] = 1
                result_holder["output"] = (
                    f"Error executing command: {type(e).__name__}: {e}"
                )
            finally:
                # 无论成功还是异常，最终都要设置事件标志，通知等待线程任务已完成
                execution_done.set()

        # ---------- 5. 启动子线程并等待完成（带超时） ----------
        # daemon=True 确保主进程退出时子线程自动终止，避免残留
        thread = threading.Thread(target=run_command, daemon=True)
        thread.start()

        # 阻塞等待子线程完成，最多等待 effective_timeout 秒
        if not execution_done.wait(timeout=effective_timeout):
            # 超时后返回错误信息，但子线程仍在后台继续运行（不强制杀进程）
            return ExecuteResponse(
                output=(
                    f"Error: Command timed out after {effective_timeout} seconds."
                ),
                exit_code=124,  # 124 是 Shell 超时的约定退出码
                truncated=False,
            )

        # ---------- 6. 从共享字典中取出执行结果 ----------
        output = result_holder.get("output", "")
        exit_code = result_holder.get("exit_code", 1)

        # ---------- 7. 输出截断（按字节长度，而非字符长度） ----------
        # 使用字节长度确保不会因多字节字符（如中文）而超限，避免截断出无效 UTF-8 序列
        truncated = False
        encoded = output.encode("utf-8")
        if len(encoded) > self._max_output_bytes:
            # 按字节截断，忽略截断处可能产生的无效 UTF-8 序列
            output = encoded[: self._max_output_bytes].decode(
                "utf-8", errors="ignore"
            )
            # 追加截断提示信息，告知用户输出不完整
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        # ---------- 8. 构造并返回响应对象 ----------
        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """通过 tar 归档包将文件上传到容器中。

        参数：
            files: `(路径, 内容)` 元组的列表，路径为容器内的目标路径，
                   若为相对路径则自动补全到工作目录下。

        返回：
            `FileUploadResponse` 列表，每个文件对应一个响应对象。
        """
        responses: list[FileUploadResponse] = []

        for file_path, content in files:
            try:
                # ---------- 1. 规范化文件路径 ----------
                # 若路径不是以 "/" 开头，则视为相对路径，拼接到工作目录下
                if not file_path.startswith("/"):
                    file_path = f"{self._working_dir}/{file_path}"

                # 提取目标文件的所在目录（若文件在根目录，则目录为 "/"）
                dir_path = file_path.rsplit("/", 1)[0] if "/" in file_path else "/"

                # ---------- 2. 构建单文件 tar 归档（内存中） ----------
                tar_stream = io.BytesIO()
                with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                    # TarInfo 的 name 不能以 "/" 开头，否则会被视为绝对路径但行为怪异
                    info = tarfile.TarInfo(name=file_path.lstrip("/"))
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))  # 将文件内容写入归档

                tar_stream.seek(0)  # 重置指针，准备读取

                # ---------- 3. 确保目标目录存在 ----------
                # 调用 execute 创建目录（mkdir -p 不会因目录已存在而报错）
                self.execute(f"mkdir -p {dir_path}")

                # ---------- 4. 将 tar 包解压到容器的根目录 "/" ----------
                # Docker SDK 的 put_archive 会将 tar 流解压到指定的容器目录
                # 因 tar 包内文件路径为 /path/to/file，解压到 "/" 后正好落在目标位置
                self._container.put_archive("/", tar_stream)

                # ---------- 5. 记录成功并添加到响应列表 ----------
                responses.append(FileUploadResponse(path=file_path, error=None))
                logger.debug("Uploaded file: %s (%d bytes)", file_path, len(content))

            except Exception as e:
                # ---------- 6. 捕获所有异常，记录错误但继续处理下一个文件 ----------
                error_msg = f"Failed to upload {file_path}: {type(e).__name__}: {e}"
                logger.error(error_msg)
                responses.append(FileUploadResponse(path=file_path, error=error_msg))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """通过 tar 归档从容器中下载文件。

        参数：
            paths: 要下载的文件路径列表，支持相对路径（自动补全到工作目录）。

        返回：
            `FileDownloadResponse` 列表，每个文件对应一个响应对象。
        """
        responses: list[FileDownloadResponse] = []

        for file_path in paths:
            try:
                # ---------- 1. 规范化文件路径 ----------
                if not file_path.startswith("/"):
                    file_path = f"{self._working_dir}/{file_path}"

                # ---------- 2. 从容器中获取文件归档流 ----------
                # get_archive 返回一个 (tar_stream, stat) 元组，其中 tar_stream 是迭代器
                tar_stream, _stat = self._container.get_archive(file_path)

                # ---------- 3. 将流式数据写入内存中的 BytesIO ----------
                tar_data = io.BytesIO()
                for chunk in tar_stream:
                    tar_data.write(chunk)
                tar_data.seek(0)  # 准备读取

                # ---------- 4. 解析 tar 归档，提取实际文件内容 ----------
                with tarfile.open(fileobj=tar_data, mode="r") as tar:
                    members = tar.getmembers()
                    # 若归档为空，说明路径可能是目录或不存在
                    if not members:
                        responses.append(
                            FileDownloadResponse(
                                path=file_path,
                                content=None,
                                error="file_not_found",
                            )
                        )
                        continue

                    # 取第一个成员（通常只有一个文件）
                    f = tar.extractfile(members[0])
                    if f is None:  # 如果成员是目录或无法提取
                        responses.append(
                            FileDownloadResponse(
                                path=file_path,
                                content=None,
                                error="file_not_found",
                            )
                        )
                        continue

                    # ---------- 5. 读取内容并记录成功 ----------
                    content = f.read()
                    responses.append(
                        FileDownloadResponse(
                            path=file_path, content=content, error=None
                        )
                    )
                    logger.debug(
                        "Downloaded file: %s (%d bytes)", file_path, len(content)
                    )

            except docker.errors.NotFound:
                # ---------- 6. 处理 Docker 返回的 404 错误（文件或目录不存在） ----------
                responses.append(
                    FileDownloadResponse(
                        path=file_path, content=None, error="file_not_found"
                    )
                )
            except Exception as e:
                # ---------- 7. 捕获其他异常，记录错误但继续处理 ----------
                error_msg = (
                    f"Failed to download {file_path}: {type(e).__name__}: {e}"
                )
                logger.error(error_msg)
                responses.append(
                    FileDownloadResponse(
                        path=file_path, content=None, error=error_msg
                    )
                )

        return responses

    def close(self) -> None:
        """关闭容器"""
        try:
            self._container.stop(timeout=5)
            if self._auto_remove:
                self._container.remove(force=True)
                logger.info("Container %s removed", self._container.short_id)
            else:
                logger.info("Container %s stopped", self._container.short_id)
        except Exception as e:
            logger.warning("Error closing container: %s", e)
        finally:
            self._client.close()

    def __enter__(self) -> DockerSandbox:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

class DockerImageNotFound(Exception):
    """镜像未发现异常"""

    def __init__(self, image: str) -> None:
        self.image = image
        super().__init__(
            f"Docker image '{image}' not found locally. "
            f"Pull it first: docker pull {image}"
        )