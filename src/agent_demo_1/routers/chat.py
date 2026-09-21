from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.concurrency import run_in_threadpool

from ..agent import AgentResult, DocumentMultiAgent
from ..logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

ChatText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]


class ChatHistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: ChatText


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: ChatText
    history: list[ChatHistoryMessage] = Field(default_factory=list)


@lru_cache
def get_document_agent() -> DocumentMultiAgent:
    return DocumentMultiAgent()


def _to_langchain_messages(
    history: list[ChatHistoryMessage],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for message in history:
        if message.role == "user":
            messages.append(HumanMessage(content=message.content))
        else:
            messages.append(AIMessage(content=message.content))
    return messages


@router.post("/chat", response_model=AgentResult)
async def post_chat(
    request: ChatRequest,
    agent: Annotated[DocumentMultiAgent, Depends(get_document_agent)],
) -> AgentResult:
    history = _to_langchain_messages(request.history)
    try:
        return await run_in_threadpool(agent.ask, request.question, history)
    except (APITimeoutError, TimeoutError) as error:
        logger.exception("chat request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="chat service timed out",
        ) from error
    except Exception as error:
        logger.exception("chat request failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="chat service failed",
        ) from error


__all__ = [
    "ChatHistoryMessage",
    "ChatRequest",
    "get_document_agent",
    "router",
]
