from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage

from tools.skills import get_skills


class SkillMiddleware(AgentMiddleware):
    """中间件，将技能描述注入到系统提示词中。"""

    def __init__(self):
        """根据 SKILLS 初始化并生成 skills 提示词。"""
        skills_list = []
        for skill in get_skills():
            skills_list.append(f"- **{skill['name']}**: {skill['description']}")
        self.skills_prompt = "\n".join(skills_list)

    def _inject_skills(self, request: ModelRequest) -> ModelRequest:
        """构造注入技能描述后的新请求。"""
        skills_addendum = (
            f"\n\n 可用的技能:\n\n{self.skills_prompt}"
            "当您需要有关处理特定类型请求的详细信息时"
            "请使用load_skill工具"
        )
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步调用：注入技能描述后透传。"""
        return handler(self._inject_skills(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """异步调用（astream/ainvoke 走这条）：注入技能描述后透传。

        流式场景（astream_events）必须实现该方法，否则会抛
        NotImplementedError，导致整个流失败。
        """
        return await handler(self._inject_skills(request))
