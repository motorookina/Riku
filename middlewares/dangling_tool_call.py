"""悬空工具调用清理中间件。

背景：LangGraph 正常流程中「模型产出 tool_calls」之后必然接着工具执行，
before_model 钩子只在「即将调用模型」前触发。因此如果此刻消息历史里存在
一条带 tool_calls、但其中某个 tool_call_id 没有对应 ToolMessage 的 AI 消息，
说明该工具调用从未执行 / 结果未落库——典型是网络中断后残留在 checkpoint 里。

这种历史直接发给 OpenAI 兼容 API（如 DeepSeek）会被 400 拒绝：
    "An assistant message with 'tool_calls' must be followed by tool messages
     responding to each 'tool_call_id'."

本中间件在每次调模型前把这些悬空的 AI 消息剪掉，保证发出去的永远是合法历史。
丢弃而非重放：中断的工具调用（如可能已部分执行的 shell 命令）重放有风险，
让模型在后续回复里自行决定要不要重新发起更安全。
"""

import uuid
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ContextT, ResponseT
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime


class DanglingToolCallMiddleware(
    AgentMiddleware[AgentState[Any], ContextT, ResponseT]
):
    """before_model：移除历史中无对应工具结果的 tool_call 残留。"""

    def _find_dangling_ids(self, state: AgentState[Any]) -> list[str]:
        """找出所有「带 tool_calls 但存在未响应 tool_call_id」的 AI 消息 id。"""
        messages = state["messages"]
        tool_ids_present = {
            m.tool_call_id for m in messages
            if isinstance(m, ToolMessage) and m.tool_call_id
        }
        dangling = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                unmatched = [
                    tc.get("id") for tc in msg.tool_calls
                    if tc.get("id") and tc["id"] not in tool_ids_present
                ]
                if unmatched:
                    if msg.id is None:
                        msg.id = str(uuid.uuid4())  # RemoveMessage 需要消息有 id
                    dangling.append(msg.id)
        return dangling

    def before_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """同步路径（agent.stream）：移除悬空 tool_call 消息。"""
        dangling_ids = self._find_dangling_ids(state)
        if not dangling_ids:
            return None
        return {"messages": [RemoveMessage(id=mid) for mid in dangling_ids]}

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """异步路径（ainvoke/astream）：与同步逻辑一致。"""
        return self.before_model(state, runtime)
