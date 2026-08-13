import asyncio
import json
import docker

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.config import get_config
from langgraph.types import Command

from config import config
from core.sandbox import DockerSandbox
from deepagents.middleware.filesystem import FilesystemMiddleware



_agent = None
MODEL=config.MODEL
LLM_API_KEY=config.LLM_API_KEY
LLM_BASE_URL=config.LLM_BASE_URL
volumes={
    "D:/Agents/db-ops-v3/win/core/workspace":{
        "bind":"/workspace",
        "mode":"rw"
    }
}
docker_sandbox=DockerSandbox(volumes=volumes)


def get_agent():
    """懒加载单例 agent，避免每次请求重建模型。"""
    global _agent
    if _agent is None:
        model = ChatDeepSeek(model=MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        _agent = create_agent(
            model=model,
            tools=[],
            middleware=[FilesystemMiddleware(backend=docker_sandbox)],
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
        )
    return _agent

# 流式输出生成器
def event_generator(user_input: str):
    agent=get_agent()
    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=["messages", "updates"]
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


if __name__ == "__main__":
    # 测试生成器
    for sse_event in event_generator("列出你所在的目录下的所有文件？"):
        print(sse_event)  # 会看到 SSE 格式的原始输出