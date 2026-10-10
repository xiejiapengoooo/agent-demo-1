from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from .agents import (
    ReviewDecision,
)
from .agents.common import Evidence, merge_evidence
from .skills.types import ExportedFile


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: NotRequired[Literal["direct", "research"]]
    skill: NotRequired[str | None]
    files: NotRequired[list[ExportedFile]]
    research_goal: NotRequired[str]
    evidence: Annotated[list[Evidence], merge_evidence]
    draft: NotRequired[str]
    review: ReviewDecision | None
    research_rounds: int
    revision_rounds: int


__all__ = ["AgentState", "ExportedFile"]
