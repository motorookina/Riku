import asyncio
import json
import uuid
from pathlib import Path
import docker

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.config import get_config
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from config import config
from core.sandbox import DockerSandbox
from deepagents.middleware.filesystem import FilesystemMiddleware
from middlewares.skills import SkillMiddleware
from tools.skills import load_skill


_agent = None
# 短期记忆：内存版 checkpointer。同一 thread_id 内累计上下文，进程退出即清空。
_memory = MemorySaver()

MODEL=config.MODEL
LLM_API_KEY=config.LLM_API_KEY
LLM_BASE_URL=config.LLM_BASE_URL
cpu_limit=config.CPU_LIMIT
memory_limit=config.MEMORY_LIMIT

# 挂载路径从 __file__ 推导，Windows/Linux 双平台通吃，避免硬编码绝对路径
_BASE_DIR = Path(__file__).resolve().parent.parent          # = win/ 根目录
_WORKSPACE_DIR = _BASE_DIR / "core" / "workspace"
_SKILLS_DIR = _BASE_DIR / "tools" / "skills"

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
    "TUSHARE_TOKEN":config.TUSHARE_TOKEN
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
            middleware=[FilesystemMiddleware(backend=docker_sandbox),SkillMiddleware()],
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
            ),
            checkpointer=_memory,
        )
    return _agent

# 流式输出生成器
def event_generator(user_input: str):
    agent=get_agent()
    # checkpointer 要求 thread_id：每个 SSE 请求用独立线程 = 无状态、互不串话
    thread_id = f"sse-{uuid.uuid4().hex}"
    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
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
    user_input: str,
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
    记忆：
        同一 session_id 在进程内累计上下文（MemorySaver 内存版）；
        不同 session_id 互相隔离；进程重启即清空。
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
    try:
        for stream_mode, chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
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
    finally:
        elapsed = time.perf_counter() - start
        write(f"{RESET}\n—— 完成，耗时 {elapsed:.1f}s ——\n")


def chat() -> None:
    """交互式命令行聊天（类似 Claude CLI 的 REPL）。

    启动后逐条接收输入，用 stream_console 流式回复（打字机效果）。
    命令：
        /exit、/quit  退出
        /clear、/new  开启新会话（清空之前上下文）
        Ctrl+C        中断当前回复（或退出）
    记忆：同一次运行内默认会话累计上下文（MemorySaver 内存版），进程重启即清空。
    """
    import sys

    ansi = sys.stdout.isatty()
    def c(code: str) -> str:
        return code if ansi else ""
    CYAN = c("\033[96m")
    GREEN = c("\033[92m")
    RESET = c("\033[0m")

    print(f"{GREEN}进入交互模式：直接输入问题，/exit 或 /quit 退出，/clear 开新会话，Ctrl+C 中断回复。{RESET}")
    session_id = "console-1"
    n = 1
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
        if text in ("/exit", "/quit"):
            break
        if text in ("/clear", "/new"):
            n += 1
            session_id = f"console-{n}"  # 换新 thread_id = 清空记忆
            print(f"{GREEN}（已开启新会话，之前的上下文已清空）{RESET}")
            continue

        try:
            stream_console(text, session_id=session_id)
        except KeyboardInterrupt:
            print("\n（已中断当前回复）")
        except Exception as e:
            print(f"\n[出错] {type(e).__name__}: {e}")


if __name__ == "__main__":
    # 交互式命令行入口
    chat()