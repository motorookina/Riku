# db-ops-v3 Linux 服务器部署文档

> 适用：把本项目（`win/` 目录下的 agent 工程）部署到 Linux 服务器。
> 文档版本：2026-08-13。若代码有变更，请同步更新本文档。

---

## 一、项目结构与架构

```
db-ops-v3/
├── linux/          # 预留，当前为空
├── win/            # 主要工程（包根目录，所有命令在此目录下执行）
│   ├── .env        # 配置 + 密钥（已被 .gitignore 忽略，勿提交）
│   ├── config.py   # 读取 .env 的配置类
│   ├── requirements.txt  # 跨平台依赖清单
│   ├── core/
│   │   ├── agent.py      # 入口：agent 构建 / 流式 / 交互式 chat
│   │   ├── sandbox.py    # Docker 沙箱
│   │   └── workspace/    # 容器 /workspace 的挂载目录（产出文件落这里）
│   ├── middlewares/      # 技能中间件
│   └── tools/
│       ├── skills.py     # load_skill 工具（host 侧读取技能目录）
│       └── skills/       # 技能资产，挂载进容器 /workspace/skills（只读）
└── DEPLOY.md        # 本文件
```

**两层运行模型**：

```
┌─ Host 层（服务器上的 Python 进程）─────────────────────┐
│  agent.py / langchain / langgraph / deepagents       │
│   · 连 Docker 管理沙箱容器                            │
│   · 连 DeepSeek API（LLM）                           │
│   · 记忆 = MemorySaver（进程内存，重启即清空）          │
└─────────────────────────────────────────────────────┘
        │ docker.from_env()（Linux 自动用 /var/run/docker.sock）
        ▼
┌─ 容器层（uv python3.13 slim 沙箱）────────────────────┐
│  /workspace          ← win/core/workspace (rw)      │
│  /workspace/skills   ← win/tools/skills (ro)        │
│  bash/read/write 工具在此执行；模型按需 pip install    │
│  · 连 Postgres（经注入的 DB_* 环境变量）               │
│  · 连 Tushare（经 TUSHARE_TOKEN）                    │
└─────────────────────────────────────────────────────┘
```

---

## 二、前置条件

- Linux 服务器（Ubuntu 20.04+ / Debian 均可），可访问外网（DeepSeek、GHCR、Tushare）
- **Docker Engine**（不是 Docker Desktop）
- PostgreSQL 可达：远程（当前 `222.73.85.26`）正常；**若在服务器本机，注意容器网络问题（见 §五.3）**

---

## 三、部署步骤

### Step 0 · 部署前必须完成的代码项

> 以下改动在**本地改好、提交**后再部署，避免在服务器上改代码。

**`env_vars` 键名统一**（强烈建议，否则 tushare/sql-pro 脚本写库会 `KeyError`）：
`.env` 键名是 `DB_PASSWD`，但容器内脚本读的是 `DB_PASSWORD`。在 `agent.py` 的 `env_vars` 里映射：
```python
env_vars = {
    "DB_USER":       config.DB_USER,
    "DB_PASSWORD":   config.DB_PASSWD,   # 键名对齐容器脚本
    "DB_HOST":       config.DB_HOST,
    "DB_PORT":       config.DB_PORT,
    "DB_NAME":       config.DB_NAME,
    "TUSHARE_TOKEN": config.TUSHARE_TOKEN,
}
```

### Step 1 · 代码落位

```bash
mkdir -p /opt/app && cd /opt/app
git clone <你的仓库地址> db-ops-v3
# 工程保持在 win/ 目录下（包结构以 win/ 为根，改名会破坏相对导入）
cd /opt/app/db-ops-v3/win
```

### Step 2 · Python 环境 + 依赖

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 安装工程依赖（requirements.txt 已整理为跨平台清单）
pip install -r requirements.txt

# 装完冻结一份干净清单，方便以后复现
pip freeze > requirements-linux.txt

# 冒烟测试：能否成功导入（会连接 Docker 并复用/创建沙箱容器）
python -c "import core.agent; print('import OK')"
```

> 若 `import` 报缺包，按报错补齐即可。

### Step 3 · 配置 `.env`

```bash
cd /opt/app/db-ops-v3/win
vim .env
chmod 600 .env    # 密钥文件只允许属主读写
```

`.env` 需要包含以下键（值来自本地 `win/.env`，按服务器环境调整）：

| 键 | 说明 | 部署时注意 |
|---|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `MODEL` | DeepSeek | 沿用 |
| `TUSHARE_TOKEN` | Tushare | 沿用（容器拉数据要用） |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWD` | Postgres | **DB_HOST 若在本机见 §五.3** |
| `_DEFAULT_IMAGE` | `ghcr.io/astral-sh/uv:python3.13-bookworm-slim` | 沿用；若用自定义镜像改成镜像名 |
| `_DEFAULT_CONTAINER_NAME` | 容器名 | 建议换个服务器专用名（如 `fin-box-linux`） |
| `_DEFAULT_WORKING_DIR` | `/workspace` | 沿用 |
| `_DEFAULT_EXECUTE_TIMEOUT` | 120 | 沿用 |
| `_DEFAULT_MAX_OUTPUT_BYTES` | 524288000 | 沿用 |
| `CPU_LIMIT` | 4 | 沿用 |
| `MEMORY_LIMIT` | "4g" | 沿用 |

> `config.py` 用 `load_dotenv()`（无参），**只在当前工作目录找 `.env`**。所以一切运行命令都要在 `win/` 目录下执行。

### Step 4 · Docker 准备

```bash
# 安装 Docker Engine（Ubuntu 为例）
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker

# 让运行用户能访问 docker socket（避免用 root 跑进程）
sudo usermod -aG docker $USER && newgrp docker

# ⚠️ 关键：镜像不会自动拉取（sandbox._check_image 只检查本地），必须手动拉
docker pull ghcr.io/astral-sh/uv:python3.13-bookworm-slim

docker info   # 验证
```

### Step 5 · 运行与验证

```bash
cd /opt/app/db-ops-v3/win && source .venv/bin/activate

# 方式 A：交互式 CLI（人工验证）
python -m core.agent

# 方式 B：作为库被你的后端调用（SSE 生成器）
# from core.agent import event_generator
# for sse in event_generator("..."): ...
```

验证清单：
1. 首次运行会**创建沙箱容器**；`docker ps` 确认挂载两项正确：
   ```bash
   docker inspect <容器名> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{println}}{{end}}'
   # 期望：/opt/app/db-ops-v3/win/core/workspace -> /workspace (rw)
   #       /opt/app/db-ops-v3/win/tools/skills  -> /workspace/skills (ro)
   ```
2. 容器内环境变量已注入：
   ```bash
   docker exec <容器名> env | grep -E 'TUSHARE|DB_'
   ```
3. 问一个涉及工具的问题，确认 `load_skill` 取到技能、`/workspace/skills` 可见。

### Step 6 ·（可选）后台常驻

交互式 chat 需要 TTY，不适合 systemd 前台，用 tmux：

```bash
tmux new -s agent
cd /opt/app/db-ops-v3/win && source .venv/bin/activate && python -m core.agent
# Ctrl+B D 脱离；tmux attach -t agent 回来
```

---

## 四、Docker 容器要点

### 1. 按名字复用 → 改配置必须 `docker rm`

`_get_or_create_container` 按容器名复用，**volumes / env / cpu / mem 只在容器创建时固化**。改了 `.env` 或 `agent.py` 里相关配置后：

```bash
docker rm -f <容器名>     # 下次运行自动按新配置重建
```

### 2. 容器依赖：按需装 vs 自定义镜像

- **现状**：slim 镜像没装 tushare/psycopg2，模型运行时会现场 `pip install`，装进容器可写层。开发 OK，**生产不稳定**（删容器就重装、可能装失败、浪费轮次）。
- **推荐（生产）**：自建镜像固化依赖：

  ```dockerfile
  # Dockerfile.agent
  FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
  RUN uv pip install --system tushare psycopg2-binary requests pandas sqlalchemy
  ```
  ```bash
  docker build -t db-ops-sandbox -f Dockerfile.agent .
  # 然后把 .env 的 _DEFAULT_IMAGE 改成 db-ops-sandbox
  ```

### 3. 网络：容器访问 DB / LLM / Tushare

- DeepSeek、Tushare 走外网，默认桥接网络正常。
- **Postgres 若在服务器本机**：默认 bridge 网络里容器的 `localhost` 是容器自己，**连不上宿主机的 Postgres**。两种解法：
  - 简单：`.env` 的 `DB_HOST` 填服务器局域网 IP；
  - 或改 `sandbox.py` 给容器加 `network_mode="host"`（容器直接用宿主机网络，`localhost` 即宿主机，但要改代码、且与"按名复用"的容器需重建配合）。

### 4. 端口 / 防火墙

本项目自身**不监听任何端口**（CLI / 库）。若你自己的后端对外提供 API，防火墙放行后端端口即可，与容器无关。

---

## 五、常见问题排查

| 现象 | 原因 / 解法 |
|---|---|
| `docker pull` 超时 / 失败 | 服务器无法访问 GHCR，配置镜像加速或代理 |
| `Error response: invalid mode: r` | volumes 的 mode 必须 `rw` / `ro`，不能写 `r` |
| `Error response: Duplicate mount point` | 两个目录绑到同一容器路径，改不同的 bind |
| 建容器时报错（`int(None*1e9)` 等） | `.env` 缺 `CPU_LIMIT` / `MEMORY_LIMIT`，config 读到 `None` |
| 改了 `.env` 不生效 | 容器按名复用，`docker rm -f <容器名>` 后重建 |
| 模型说"找不到技能/脚本" | 确认 `/workspace/skills` 挂载成功且为 `ro`；SKILL.md 内为相对路径，可能需补 `/workspace/skills/<skill>/` 前缀 |
| tushare 报"token 缺失/无权限" | 容器内无 `TUSHARE_TOKEN`（重建容器时注入）或 token 无积分权限 |
| 写库报 `KeyError: 'DB_PASSWORD'` | 未做 §三.Step 0 的键名统一，`.env` 是 `DB_PASSWD` |
| 中文/emoji 在终端乱码或崩溃 | Linux 默认 UTF-8，本项目已有编码守卫，一般无此问题 |
