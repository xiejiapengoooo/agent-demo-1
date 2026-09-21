from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from .common import last_user_text

SUPERVISOR_PROMPT = """你是文档问答系统的监督者，只负责选择处理路径。

选择 direct 的情况：打招呼、询问系统能力、无需项目知识库即可完成的普通交流。
选择 research 的情况：问题涉及知识库中的事实、价格、规则、政策、流程，或者是对这些问题的追问。

如果选择 research，请把用户问题和必要的对话上下文改写成一个独立、明确的 research_goal。
不要回答用户问题。"""


class RouteDecision(BaseModel):
    route: Literal["direct", "research"]
    research_goal: str | None = None
    reason: str = ""


class SupervisorAgent:
    def __init__(self, model: BaseChatModel) -> None:
        self._router = model.with_structured_output(
            RouteDecision,
            method="function_calling",
        )

    def invoke(self, messages: Sequence[BaseMessage]) -> RouteDecision:
        raw_decision = self._router.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), *messages]
        )
        decision = RouteDecision.model_validate(raw_decision)
        if decision.route == "research" and not (decision.research_goal or "").strip():
            decision = decision.model_copy(
                update={"research_goal": last_user_text(messages)}
            )
        return decision


__all__ = ["RouteDecision", "SupervisorAgent"]
