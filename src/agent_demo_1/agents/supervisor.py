import json
from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ..skills import SkillRegistry
from .base import BaseAgent
from .common import last_user_text

SUPERVISOR_PROMPT = """你是系统监督者，只负责选择处理路径。

选择 direct 的情况：打招呼、询问系统能力、无需项目知识库即可完成的普通交流。
选择 research 的情况：问题涉及知识库中的事实、价格、规则、政策、流程，或者是对这些问题的追问。

如果选择 research，请把用户问题和必要的对话上下文改写成一个独立、明确的 research_goal。
根据用户当前请求判断是否需要执行下方技能，历史消息仅用于理解上下文。
需要执行时设置 skill 为技能名称；否则设置 skill 为 null。
只能选择技能目录中存在的名称。执行技能时选择 direct 路径。
不要回答用户问题。"""


class RouteDecision(BaseModel):
    route: Literal["direct", "research"]
    research_goal: str | None = None
    reason: str = ""
    skill: str | None = None


class SupervisorAgent(BaseAgent):
    name = "supervisor"
    system_prompt = SUPERVISOR_PROMPT

    def __init__(
        self, model: BaseChatModel, *, skills: SkillRegistry | None = None
    ) -> None:
        super().__init__(model)
        self._skills = skills
        if self._skills is not None:
            catalog = [
                skill.model_dump(mode="json") for skill in self._skills.catalog()
            ]
            self.system_prompt = (
                f"{SUPERVISOR_PROMPT}\n\n【可用技能】\n"
                f"{json.dumps(catalog, ensure_ascii=False)}"
            )
        self._router = self.model.with_structured_output(
            RouteDecision,
            method="function_calling",
            extra_body={"thinking": {"type": "disabled"}},
        )

    def invoke(self, messages: Sequence[BaseMessage]) -> RouteDecision:
        raw_decision = self._router.invoke(self._with_system_prompt(messages))
        decision = RouteDecision.model_validate(raw_decision)
        if decision.skill is not None and self._skills is not None:
            self._skills.get(decision.skill)
            return decision.model_copy(
                update={"route": "direct", "research_goal": None}
            )
        if decision.route == "research" and not (decision.research_goal or "").strip():
            decision = decision.model_copy(
                update={"research_goal": last_user_text(messages)}
            )
        return decision


__all__ = ["RouteDecision", "SupervisorAgent"]
