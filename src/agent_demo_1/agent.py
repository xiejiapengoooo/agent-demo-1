from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import SecretStr

from .agents import (
    AgentResult,
    AnswererAgent,
    Evidence,
    ResearcherAgent,
    ReviewDecision,
    ReviewerAgent,
    RouteDecision,
    Source,
    SupervisorAgent,
)
from .agents.common import last_user_text, merge_evidence
from .config import Settings, get_settings


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: NotRequired[Literal["direct", "research"]]
    research_goal: NotRequired[str]
    evidence: Annotated[list[Evidence], merge_evidence]
    draft: NotRequired[str]
    review: ReviewDecision | None
    research_rounds: int
    revision_rounds: int


def _default_model(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model=settings.agent_model,
        base_url=settings.openai_base_url,
        temperature=0,
        timeout=settings.agent_timeout,
        max_retries=2,
        api_key=SecretStr(os.environ["SU8_API_KEY"]),
    )


class DocumentMultiAgent:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        shared_model = _default_model(self.settings)
        self._supervisor = SupervisorAgent(shared_model)
        self._researcher = ResearcherAgent(shared_model)
        self._answerer = AnswererAgent(shared_model)
        self._reviewer = ReviewerAgent(shared_model)
        self._graph = self._build_graph()

    def ask(
        self,
        question: str,
        history: Sequence[BaseMessage] = (),
    ) -> AgentResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if any(not isinstance(message, BaseMessage) for message in history):
            raise TypeError("history must contain LangChain messages")

        initial_state: AgentState = {
            "messages": [*history, HumanMessage(content=question.strip())],
            "evidence": [],
            "research_rounds": 0,
            "revision_rounds": 0,
            "review": None,
        }
        state = self._graph.invoke(
            initial_state,
            config={"recursion_limit": self.settings.agent_recursion_limit},
        )

        answer = state.get("draft", "").strip()
        if not answer:
            raise RuntimeError("agent returned an empty answer")
        route = cast(Literal["direct", "research"], state["route"])
        sources = self._cited_sources(answer, state.get("evidence", []))
        return AgentResult(answer=answer, sources=sources, route=route)

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("answerer", self._answerer_node)
        builder.add_node("reviewer", self._reviewer_node)
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {"direct": "answerer", "research": "researcher"},
        )
        builder.add_edge("researcher", "answerer")
        builder.add_conditional_edges(
            "answerer",
            self._route_after_answer,
            {"review": "reviewer", "end": END},
        )
        builder.add_conditional_edges(
            "reviewer",
            self._route_after_review,
            {
                "pass": END,
                "revise": "answerer",
                "more_research": "researcher",
            },
        )
        return builder.compile()

    def _supervisor_node(self, state: AgentState) -> dict[str, Any]:
        decision = self._supervisor.invoke(state["messages"])
        return {
            "route": decision.route,
            "research_goal": (decision.research_goal or "").strip(),
        }

    def _researcher_node(self, state: AgentState) -> dict[str, Any]:
        review = state.get("review")
        goal = state.get("research_goal") or last_user_text(state["messages"])
        if review is not None and review.verdict == "more_research":
            goal = (review.missing_query or review.feedback or goal).strip()

        evidence = self._researcher.invoke(
            goal,
            existing_evidence=state.get("evidence", []),
        )
        return {
            "evidence": evidence,
            "research_rounds": state.get("research_rounds", 0) + 1,
            "review": None,
        }

    def _answerer_node(self, state: AgentState) -> dict[str, Any]:
        review = state.get("review")
        review_feedback = None
        revision_rounds = state.get("revision_rounds", 0)
        if review is not None and review.verdict == "revise":
            review_feedback = review.feedback
            revision_rounds += 1

        answer = self._answerer.invoke(
            state["messages"],
            route=self._route_after_supervisor(state),
            evidence=state.get("evidence", []),
            review_feedback=review_feedback,
        )
        return {
            "draft": answer,
            "revision_rounds": revision_rounds,
            "review": None,
        }

    def _reviewer_node(self, state: AgentState) -> dict[str, Any]:
        draft = state.get("draft")
        if draft is None:
            raise RuntimeError("answerer did not provide a draft")
        decision = self._reviewer.invoke(
            question=last_user_text(state["messages"]),
            draft=draft,
            evidence=state.get("evidence", []),
        )
        return {"review": decision}

    @staticmethod
    def _route_after_supervisor(
        state: AgentState,
    ) -> Literal["direct", "research"]:
        route = state.get("route")
        if route is None:
            raise RuntimeError("supervisor did not provide a route")
        return route

    @staticmethod
    def _route_after_answer(state: AgentState) -> Literal["review", "end"]:
        route = DocumentMultiAgent._route_after_supervisor(state)
        return "end" if route == "direct" else "review"

    def _route_after_review(
        self,
        state: AgentState,
    ) -> Literal["pass", "revise", "more_research"]:
        review = state.get("review")
        if review is None or review.verdict == "pass":
            return "pass"
        if review.verdict == "revise":
            if (
                state.get("revision_rounds", 0)
                < self.settings.agent_max_revision_rounds
            ):
                return "revise"
            return "pass"
        if state.get("research_rounds", 0) < self.settings.agent_max_research_rounds:
            return "more_research"
        return "pass"

    @staticmethod
    def _cited_sources(answer: str, evidence: Sequence[Evidence]) -> list[Source]:
        sources = []
        for item in evidence:
            if f"[{item.chunk_id}]" not in answer:
                continue
            sources.append(Source.model_validate(item.model_dump(exclude={"text"})))
        return sources


__all__ = [
    "AgentResult",
    "DocumentMultiAgent",
    "Evidence",
    "ReviewDecision",
    "RouteDecision",
    "Source",
]
