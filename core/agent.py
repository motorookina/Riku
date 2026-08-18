import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
import docker

import httpx
import openai
import psycopg
from psycopg.rows import dict_row

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.config import get_config
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from config import config
from core.sandbox import DockerSandbox
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import SummarizationMiddleware
from middlewares.dangling_tool_call import DanglingToolCallMiddleware
from middlewares.planning import PlanningMiddleware
from middlewares.skills import SkillMiddleware
from tools.skills import load_skill

logger = logging.getLogger(__name__)

_agent = None
_checkpointer = None   # 持久化 checkpointer（懒加载），见 _get_checkpointer()

MODEL=config.MODEL
LLM_API_KEY=config.LLM_API_KEY
LLM_BASE_URL=config.LLM_BASE_URL
cpu_limit=config.CPU_LIMIT
memory_limit=config.MEMORY_LIMIT

# 挂载路径从 __file__ 推导，Windows/Linux 双平台通吃，避免硬编码绝对路径
_BASE_DIR = Path(__file__).resolve().parent.parent          # = 项目根目录
_WORKSPACE_DIR = _BASE_DIR / "core" / "workspace"
_SKILLS_DIR = _BASE_DIR / "tools" / "skills"

# 会话指针：记录当前 thread_id / 计数 / 是否被中断，供跨进程 /resume
_STATE_DIR = _BASE_DIR / ".state"
_STATE_FILE = _STATE_DIR / "session_state.json"


def _load_session_state() -> dict:
    """读取会话指针（文件不存在/损坏时返回空 dict）。"""
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_session_state(state: dict) -> None:
    """写会话指针；失败不阻塞对话（续跑只是尽力而为）。"""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _get_checkpointer():
    """懒加载 checkpointer。

    .env 配置了 CHECKPOINT_DB_NAME 时用 PostgresSaver（跨进程持久化，支持断点续跑）；
    否则回退 MemorySaver（现状：进程退出即清空）。
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    db_name = config.CHECKPOINT_DB_NAME
    if not db_name:
        _checkpointer = MemorySaver()
        logger.info("CHECKPOINT_DB_NAME 未配置，回退 MemorySaver（会话不跨进程持久化）")
        return _checkpointer
    try:
        conn = psycopg.connect(
            host=config.CHECKPOINT_DB_HOST,
            port=config.CHECKPOINT_DB_PORT,
            dbname=db_name,
            user=config.CHECKPOINT_DB_USER,
            password=config.CHECKPOINT_DB_PASSWD,
            autocommit=True,          # PostgresSaver 要求：setup() 能正常提交建表
            row_factory=dict_row,     # PostgresSaver 要求：按列名取行
            connect_timeout=5,
            # 长连接保活：进程常驻，连接空闲过久可能被服务器/中间网络断开
            # （报 OperationalError: consuming input failed / server closed the connection）。
            # keepalives 是 libpq 客户端侧参数，定期发探测包防止静默断连。
            keepalives=1,
            keepalives_idle=120,      # 空闲 120s 后开始发送保活探测
            keepalives_interval=15,   # 每 15s 探测一次
            keepalives_count=5,       # 连续 5 次无响应即判定连接已死
        )
        cp = PostgresSaver(conn)
        cp.setup()  # 首次自动建表（幂等）
        _checkpointer = cp
        logger.info("已启用 Postgres 持久化 checkpoint（db=%s）", db_name)
        return cp
    except Exception as e:
        raise RuntimeError(
            f"无法连接 Postgres 持久化 checkpoint 库 {db_name}@{config.CHECKPOINT_DB_HOST}:"
            f"{config.CHECKPOINT_DB_PORT}：{type(e).__name__}: {e}\n"
            "请先在 PG 上创建该库（如 CREATE DATABASE agent_checkpoint;）并核对 CHECKPOINT_DB_* 配置；"
            "若想禁用持久化，把 .env 的 CHECKPOINT_DB_NAME 留空即可回退内存版。"
        ) from e

volumes={
    _WORKSPACE_DIR.as_posix():{
        "bind":"/workspace",
        "mode":"rw"
    },
    _SKILLS_DIR.as_posix():{
        "bind":"/workspace/skills",
        "mode":"ro"
    }# 挂载skills目录到容器指定目录下，防止agent找不到对应skill
}

env_vars={
    "DB_USER": config.DB_USER,
    "DB_PASSWD": config.DB_PASSWD,
    "DB_HOST": config.DB_HOST,
    "DB_PORT":config.DB_PORT,
    "DB_NAME": config.DB_NAME,
    "TUSHARE_TOKEN":config.TUSHARE_TOKEN,
    "LIXINGER_TOKEN":config.LIXINGER_TOKEN
}

docker_sandbox = DockerSandbox(volumes=volumes,env_vars=env_vars,cpu_limit=cpu_limit,memory_limit=memory_limit)


def get_agent():
    """懒加载单例 agent，避免每次请求重建模型。"""
    global _agent
    if _agent is None:
        model = ChatDeepSeek(model=MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        _agent = create_agent(
            model=model,
            tools=[load_skill],
            middleware=[FilesystemMiddleware(backend=docker_sandbox),SkillMiddleware(),
            PlanningMiddleware(),
            # 先清理悬空 tool_call，再让 SummarizationMiddleware 处理摘要，
            # 避免把损坏历史发给模型（DeepSeek 会以 400 拒绝）
            DanglingToolCallMiddleware(),
            SummarizationMiddleware(
                # 复用主模型实例做摘要：传字符串("openai:gpt-4o-mini")会要求装 langchain-openai +
                # OPENAI_API_KEY + 能直连 OpenAI，本环境均不具备
                model=model,
                max_tokens_before_summary=100000,  # 在 100000 个 token 时触发摘要
                messages_to_keep=100,  # 摘要后保留最近 100 条消息，适配长线任务
            )],
            system_prompt=(
                "你是可以使用各种工具帮助用户完成各种任务的助手。\n"
                "\n"
                "【安全铁律】\n"
                "1. 绝对禁止在任何文本中暴露敏感信息，包括但不限于：数据库连接串"
                "（主机/端口/库名/账号/密码）、Tushare token、API Key、JWT 密钥、"
                "以及 .env 等配置文件的内容。思考过程、工具参数、回复文本均适用。\n"
                "2. 所有敏感凭据已封装在工具内部，调用工具时不要传入、展示或推测它们；"
                "不要尝试读取 .env 或环境变量来获取密钥，也不要让用户看到这些信息。\n"
                "3. 涉及金融数据的下载与存储时，优先使用 finance_db 数据库。"
                "\n"
                "\n"
                "【任务规划】\n"
                "1. 多步骤任务开始前，先调用 todo_write 规划任务清单，把大目标拆成"
                "可执行的步骤（pending 待办 / in_progress 进行中 / completed 已完成）。\n"
                "2. 执行过程中每完成一步或切换步骤，就调用 todo_write 更新状态，按清单推进。\n"
                "3. 单步简单任务不需要规划，直接执行。\n"
                "\n"
                "\n"
                "【效率铁律】（数据下载/更新任务尤其遵守，详细规范见 data-workflow 技能）\n"
                "1. 先查已有再下载：动手前确认目标表/文件的最新日期与覆盖范围，只补缺失，"
                "默认增量更新，绝不整表重拉。\n"
                "2. 复用已有产物：/workspace 与 data/ 里已有的文件/列表/脚本直接复用，不重复下载或生成。\n"
                "3. 批量一次到位：批量接口下载、批量写入库，禁止逐条循环调工具；"
                "能一个脚本跑完的绝不拆成多次。\n"
                "4. 数据不进对话：大结果直接写库/写文件，工具只返回摘要（如「写入 N 行，最新日期 X」）。\n"
                "5. 写库幂等可重跑：按日期 upsert，重复执行不产生脏数据。\n"
                "6. 达到终点即停：完成标准明确（如「已更新到 X，写入 N 行」），达到即停，不重复校验。\n"
                "【语言限制】"
                "包括以上所有内容的所有说明性文本，推理思考过程，结论性文本，代码注释等都必须使用中文，只有必要场景比如执行命令，编写代码等才使用英文"
            ),
            checkpointer=_get_checkpointer(),  # Postgres 持久化（未配置则回退内存）
        )
    return _agent

# 流式输出生成器
def event_generator(user_input: str | None = None, session_id: str | None = None):
    agent=get_agent()
    # checkpointer 要求 thread_id：传入 session_id 可复用该线程（配合续跑）；
    # 缺省每次请求用独立线程 = 无状态、互不串话
    thread_id = session_id or f"sse-{uuid.uuid4().hex}"
    # user_input=None 表示从 checkpoint 续跑（stream(None) 语义）
    input_data = (
        None if user_input is None
        else {"messages": [{"role": "user", "content": user_input}]}
    )
    for stream_mode, chunk in agent.stream(
        input_data,
        stream_mode=["messages", "updates"],
        config={"configurable": {"thread_id": thread_id}},
    ):
        if stream_mode == "messages":
            token, meta = chunk
            # 直接发送 content_blocks 列表，让前端解析
            event = {
                "type": "message_chunk",
                "blocks": token.content_blocks,  # 列表，如 [{"type":"text","text":"你好"}, ...]
                "node": meta.get("langgraph_node")  # 可选，标识来自哪个节点
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def stream_console(
    user_input: str | None = None,
    *,
    session_id: str = "console",
    char_delay: float = 0.02,
    show_reasoning: bool = True,
    show_tool_summaries: bool = True,
) -> None:
    """以打字机效果在控制台流式输出 agent 的完整回复。

    控制台调试专用入口（event_generator 是给前端喂 SSE 的，这个给人看）：
      - 思考过程（reasoning）灰色逐字显现，最终回复以打字机效果打出
      - 工具调用 / 工具结果只给一行摘要，避免整段 skill 文档刷屏
      - 输出重定向到文件时自动去掉 ANSI 颜色

    用法：
        python -m core.agent                       # __main__ 已改为走这里
        stream_console("现在有哪些上市公司？整理成 markdown 文档")
    续跑：
        stream_console(None, session_id="console-1")  # 从 checkpoint 继续上次被中断的回复
    记忆：
        同一 session_id 累计上下文（Postgres 持久化时跨进程保留）；
        不同 session_id 互相隔离。
    """
    import sys
    import time

    agent = get_agent()

    # 只在真实终端里输出 ANSI 颜色，重定向/管道时保持纯文本
    ansi = sys.stdout.isatty()
    def c(code: str) -> str:
        return code if ansi else ""
    DIM = c("\033[90m")     # 灰色：推理
    CYAN = c("\033[96m")    # 青色：工具调用
    GREEN = c("\033[92m")   # 绿色：工具结果摘要
    YELLOW = c("\033[93m")  # 黄色：任务清单
    RESET = c("\033[0m")

    def write(s: str) -> None:
        """写 stdout 并立即刷新。

        控制台是 GBK 等窄编码时，遇到无法编码的字符（如 emoji）用 ? 替换，
        避免像 print() 那样直接抛 UnicodeEncodeError 崩溃。
        """
        try:
            sys.stdout.write(s)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            sys.stdout.write(s.encode(enc, "replace").decode(enc))
        sys.stdout.flush()

    def typewrite(text: str) -> None:
        """逐字打出，形成打字机效果。"""
        for ch in text:
            write(ch)
            time.sleep(char_delay)

    last_node = None
    tool_name = None               # 最近一次模型发起的工具名，用于标注工具结果
    tools_summary_printed = False  # 同一段工具结果只打一次摘要，避免刷屏

    start = time.perf_counter()
    completed = False
    try:
        # user_input=None → 续跑（LangGraph 的 stream(None) 语义，从最后 checkpoint 继续）
        input_data = (
            None if user_input is None
            else {"messages": [{"role": "user", "content": user_input}]}
        )
        for stream_mode, chunk in agent.stream(
            input_data,
            stream_mode=["messages", "updates"],
            config={"configurable": {"thread_id": session_id}},
        ):
            if stream_mode != "messages":
                continue
            token, meta = chunk
            node = meta.get("langgraph_node")

            # 节点切换（model <-> tools）时换行分隔，进入 tools 时重置摘要开关
            if node != last_node:
                write("\n")
                if node == "tools":
                    tools_summary_printed = False
                last_node = node

            for block in token.content_blocks or []:
                btype = block.get("type")

                # tools 节点 = 工具执行结果：只摘第一行，不整篇刷屏
                if node == "tools":
                    if (
                        btype == "text"
                        and show_tool_summaries
                        and not tools_summary_printed
                    ):
                        text = (block.get("text") or "").strip()
                        if text:
                            if tool_name == "todo_write":
                                # 任务清单整段展示（s05 同款黄色横幅），不截断
                                write(f"\n{YELLOW}## 当前任务清单\n{text}{RESET}\n")
                            else:
                                first = text.splitlines()[0][:120]
                                label = f"[工具结果] {tool_name}: " if tool_name else "[工具结果] "
                                write(f"{GREEN}{label}{first}{RESET}\n")
                            tools_summary_printed = True
                    continue

                # —— model 节点的内容块 ——
                if btype == "reasoning":
                    if show_reasoning:
                        write(f"{DIM}{block.get('reasoning', '')}{RESET}")
                elif btype in ("text", "text_delta"):
                    typewrite(block.get("text") or "")
                elif btype == "tool_call_chunk":
                    if block.get("name"):
                        tool_name = block["name"]
                        if show_tool_summaries:
                            write(f"\n{CYAN}[调用工具] {tool_name} ...{RESET}")
                elif btype == "tool_call":
                    if block.get("name"):
                        tool_name = block["name"]
                    if show_tool_summaries:
                        args = block.get("args") or ""
                        write(f"\n{CYAN}[调用工具] {tool_name} {args}{RESET}")
        completed = True
    finally:
        # 只有正常结束才打印耗时横幅；异常/中断由上层（run_turn/chat）提示
        if completed:
            elapsed = time.perf_counter() - start
            write(f"{RESET}\n—— 完成，耗时 {elapsed:.1f}s ——\n")


# 网络类异常自动重试（有界）：连接错误 / 传输层断连 / 超时 / 限流(429) / 5xx；4xx（如 400）不重试
_MAX_RETRIES = 3
_RETRY_BACKOFF = (5, 15, 30)  # 秒
# checkpoint 数据库连接断开（长连接过期）后，最多自动重连几次
_MAX_RECONNECTS = 2


def _reset_agent() -> None:
    """清空缓存的 agent 与 checkpointer，下次 get_agent() 会重建（含新的数据库连接）。

    checkpoint 连接失效后调用：会话状态都在 Postgres checkpoint 里，重建不丢上下文。
    """
    global _agent, _checkpointer
    _agent = None
    _checkpointer = None


def _checkpoint_connection_ok() -> bool:
    """检查当前 checkpoint 数据库连接是否可用；不可用返回 False（由调用方重建）。

    进程常驻时连接可能被服务器/中间网络断开（报 OperationalError: consuming input failed）。
    每次对话前做一次轻量探活，过期连接在提交用户输入之前就被发现并重建，
    避免把输入打进已断开的连接、再事后补救。
    """
    if _checkpointer is None:
        return True  # 尚未建立，get_agent() 懒加载时再连
    try:
        if isinstance(_checkpointer, PostgresSaver):
            conn = _checkpointer.conn
            if conn.closed:
                return False
            conn.cursor().execute("SELECT 1")
        return True
    except Exception:
        return False


def _is_retryable(e: Exception) -> bool:
    """是否值得自动重试的网络类异常。"""
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    if isinstance(e, openai.APIStatusError) and 500 <= e.status_code < 600:
        return True
    # 流式场景下，httpx 传输层异常（如服务端中途断连 RemoteProtocolError）不会被
    # openai SDK 包装成 APIConnectionError，而是原样抛出，同样值得重试
    if isinstance(e, httpx.TransportError):
        return True
    # 注意：4xx 属于请求本身的问题（如 400 消息历史非法），确定性地重试必然还失败，
    # 不在此重试；BadRequestError 等由上层直接报错（详见 _find_dangling_ids 的清理逻辑）
    return False


def run_turn(user_input: str | None, session_id: str) -> bool:
    """执行一轮对话；网络类异常自动重试（有界），重试用尽交回用户手动恢复。

    返回 True 表示正常结束；False 表示出错/重试用尽。
    重试时传 None（从 checkpoint 续跑），不复发原消息——原输入已在 checkpoint 中。
    """
    attempt = 0
    while True:
        try:
            stream_console(user_input if attempt == 0 else None, session_id=session_id)
            return True
        except KeyboardInterrupt:
            raise  # 交回 chat() 标记 interrupted
        except Exception as e:
            if not _is_retryable(e):
                print(f"\n[出错] {type(e).__name__}: {e}")
                return False
            if attempt >= _MAX_RETRIES:
                print(
                    f"\n（网络异常连续重试 {_MAX_RETRIES} 次仍失败。上下文已持久化，"
                    f"输入 /resume 可从中断处继续，或直接输入新问题。）"
                )
                print(f"[出错] {type(e).__name__}: {e}")
                return False
            delay = _RETRY_BACKOFF[attempt]
            print(f"\n（网络波动：{type(e).__name__}，{delay}s 后自动重试第 {attempt + 1}/{_MAX_RETRIES} 次…）")
            time.sleep(delay)
            attempt += 1


def chat() -> None:
    """交互式命令行聊天（类似 Claude CLI 的 REPL）。

    启动后逐条接收输入，用 stream_console 流式回复（打字机效果）。
    命令：
        /exit、/quit  退出
        /clear、/new  开启新会话（清空之前上下文）
        /resume       继续上次被中断的回复（上下文已持久化，跨进程有效）
        Ctrl+C        中断当前回复（或退出）
    记忆：同一 session_id 累计上下文；配置 Postgres 后跨进程持久化，支持断点续跑。
    """
    import sys

    ansi = sys.stdout.isatty()
    def c(code: str) -> str:
        return code if ansi else ""
    CYAN = c("\033[96m")
    GREEN = c("\033[92m")
    YELLOW = c("\033[93m")
    RESET = c("\033[0m")

    # 恢复会话指针：thread_id / 计数 / 上次是否被中断
    state = _load_session_state()
    n = int(state.get("counter") or 1)
    session_id = f"console-{n}"
    interrupted = state.get("last_status") not in (None, "done")

    print(f"{GREEN}进入交互模式：直接输入问题，/q 退出，/clear 开新会话，/resume 继续上次中断，Ctrl+C 中断回复。{RESET}")
    if interrupted:
        print(
            f"{YELLOW}（检测到上次会话 {session_id} 未正常结束，输入 /resume 可从中断处继续；"
            f"上下文已持久化不会丢）{RESET}"
        )

    while True:
        try:
            user_input = input(f"\n{CYAN}> {RESET}")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n再见")
            break

        text = user_input.strip()
        if not text:
            continue
        if text in ("/q"):
            break
        if text == "/resume":
            if interrupted:
                print(f"{CYAN}（正在从中断处继续…）{RESET}")
                try:
                    ok = run_turn(None, session_id)
                except KeyboardInterrupt:
                    print("\n（已中断当前回复）")
                    continue  # 状态保持 interrupted，下次仍可 /resume
                interrupted = not ok
                _save_session_state(
                    {"session_id": session_id, "counter": n,
                     "last_status": "done" if ok else "interrupted"}
                )
            else:
                print("（没有可恢复的未完成回复）")
            continue
        if text in ("/clear", "/new"):
            n += 1
            session_id = f"console-{n}"  # 换新 thread_id = 清空记忆
            interrupted = False
            _save_session_state(
                {"session_id": session_id, "counter": n, "last_status": "done"}
            )
            print(f"{GREEN}（已开启新会话 {session_id}，之前的上下文已清空）{RESET}")
            continue

        # 普通一轮：标记进行中 → run_turn（含自动重试）→ 依结果标记 done / interrupted
        try:
            _save_session_state(
                {"session_id": session_id, "counter": n, "last_status": "in_progress"}
            )
            ok = run_turn(text, session_id)
        except KeyboardInterrupt:
            print("\n（已中断当前回复）")
            ok = False
        interrupted = not ok
        _save_session_state(
            {"session_id": session_id, "counter": n,
             "last_status": "done" if ok else "interrupted"}
        )


if __name__ == "__main__":
    # 交互式命令行入口
    chat()