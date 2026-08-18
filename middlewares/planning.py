"""任务规划中间件：把 s05 TodoWrite 的「计划-执行」机制移植到 LangGraph agent。

核心机制（取自 s05_todo_write.py）：
  1. todo_write 工具：多步任务开始前，模型用它建立任务清单并随进度更新
     （每项 pending / in_progress / completed，同一时刻最多一项进行中）。
  2. 结构化任务状态：TodoManager 按会话（thread_id）隔离保存清单，渲染成文本。

不设强制提醒：是否规划、按什么节奏更新任务清单，由模型依据系统提示词自行把握，
只要大致按任务进度表推进即可。
"""

from collections import OrderedDict
from typing import Annotated, Literal, TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langgraph.config import get_config


# ---------------------------------------------------------------------------
# s05 移植：结构化任务清单（TodoManager）
# ---------------------------------------------------------------------------

class TodoItem(TypedDict):
    """单条任务：内容 + 状态。"""
    content: Annotated[str, "任务内容"]
    status: Annotated[
        Literal["pending", "in_progress", "completed"],
        "任务状态：pending 待办 / in_progress 进行中 / completed 已完成",
    ]


class TodoManager:
    """任务清单的存储与校验（s05 TodoManager 移植）。

    与 s05 一致：最多 20 项、状态必须是 pending/in_progress/completed、
    同一时刻最多一项 in_progress。
    """
    MAX_ITEMS = 20

    def __init__(self) -> None:
        self.items: list[dict] = []

    def update(self, todos: list | str) -> str:
        """校验并保存新的任务清单，返回渲染后的文本（供模型/用户查看）。"""
        if isinstance(todos, str):
            import json
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError as e:
                raise ValueError("todos 必须是任务列表（或 JSON 数组字符串）") from e

        if not isinstance(todos, list):
            raise ValueError("todos 必须是任务列表")
        if len(todos) > self.MAX_ITEMS:
            raise ValueError(f"任务最多 {self.MAX_ITEMS} 项")

        validated: list[dict] = []
        in_progress = 0
        for index, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"todos[{index}] 必须是对象")
            content = str(todo.get("content", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            if not content:
                raise ValueError(f"todos[{index}] 缺少 content")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"todos[{index}] 状态不合法: {status}")
            if status == "in_progress":
                in_progress += 1
            validated.append({"content": content, "status": status})

        if in_progress > 1:
            raise ValueError("同一时刻只能有一项 in_progress")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """把任务清单渲染成多行文本。"""
        if not self.items:
            return "（暂无任务清单）"
        markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[√]"}
        lines = [f"{markers[t['status']]} {t['content']}" for t in self.items]
        done = sum(t["status"] == "completed" for t in self.items)
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


# 每个会话（thread_id）一个 TodoManager，避免不同会话互相串计划；
# 会话数封顶，防止长驻进程内存无限增长（OrderedDict 按最近使用排序淘汰最旧）。
_MAX_THREADS = 20
_TODO_MANAGERS: "OrderedDict[str, TodoManager]" = OrderedDict()


def _todo_manager(thread_id: str) -> TodoManager:
    manager = _TODO_MANAGERS.get(thread_id)
    if manager is None:
        manager = TodoManager()
        _TODO_MANAGERS[thread_id] = manager
        if len(_TODO_MANAGERS) > _MAX_THREADS:
            _TODO_MANAGERS.popitem(last=False)  # 淘汰最久未使用的会话
    else:
        _TODO_MANAGERS.move_to_end(thread_id)
    return manager


def _current_thread_id() -> str:
    """工具运行时从 LangGraph 运行时配置取当前会话的 thread_id。"""
    cfg = get_config()
    return (cfg.get("configurable") or {}).get("thread_id") or "default"


@tool
def todo_write(todos: list[TodoItem]) -> str:
    """创建或更新当前会话的任务清单。

    多步骤任务开始前先调用本工具把大目标拆成可执行步骤并规划顺序；
    执行过程中随时调用它更新各步骤状态（pending 待办 / in_progress 进行中 /
    completed 已完成），让模型和用户始终清楚当前进度。
    返回渲染后的任务清单。
    """
    try:
        return _todo_manager(_current_thread_id()).update(todos)
    except ValueError as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 中间件：把 todo_write 工具挂进 agent
# ---------------------------------------------------------------------------

class PlanningMiddleware(AgentMiddleware):
    """把 todo_write 工具注册进 agent。

    通过 middleware.tools 挂载（create_agent 会自动合并进模型可用的工具列表），
    让模型能建立并维护任务清单；是否调用、按什么节奏更新，由模型按系统提示词
    自行把握，不做强制提醒。
    """

    def __init__(self) -> None:
        super().__init__()
        self.tools = [todo_write]
