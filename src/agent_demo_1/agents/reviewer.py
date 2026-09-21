from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .base import BaseAgent
from .common import Evidence, format_evidence

REVIEWER_PROMPT = """你是答案审核员，只负责审核草稿，不要重写答案。

检查：
1. 是否回答了用户问题；
2. 知识库事实是否都有引用；
3. 引用的证据是否真的支持对应结论；
4. 是否存在超出证据的推测或伪造；
5. 引用格式是否为 [S-xxxxxxxx]。

verdict 规则：
- pass：答案准确且证据充分；
- revise：现有证据充分，但答案表达、引用或完整性需要修改；
- more_research：缺少回答所需的证据，并在 missing_query 中给出明确的补充检索问题。"""


class ReviewDecision(BaseModel):
    verdict: Literal["pass", "revise", "more_research"]
    feedback: str = ""
    missing_query: str | None = None


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    system_prompt = REVIEWER_PROMPT

    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(model)
        self._reviewer = self.model.with_structured_output(
            ReviewDecision,
            method="function_calling",
        )

    def invoke(
        self,
        *,
        question: str,
        draft: str,
        evidence: Sequence[Evidence],
    ) -> ReviewDecision:
        prompt = (
            f"用户问题：{question}\n\n"
            f"答案草稿：\n{draft}\n\n"
            f"可用证据：\n{format_evidence(evidence)}"
        )
        raw_decision = self._reviewer.invoke(
            self._with_system_prompt([HumanMessage(content=prompt)])
        )
        return ReviewDecision.model_validate(raw_decision)


__all__ = ["ReviewDecision", "ReviewerAgent"]
