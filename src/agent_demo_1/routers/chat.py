from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated, Any, Literal
from uuid import uuid4

import anyio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..agent import AgentResult, DocumentMultiAgent
from ..events import AgentEvent
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
        return await anyio.to_thread.run_sync(agent.ask, request.question, history)
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


@router.post("/chat/stream", response_class=EventSourceResponse)
async def post_chat_stream(
    request: ChatRequest,
    agent: Annotated[DocumentMultiAgent, Depends(get_document_agent)],
) -> AsyncIterator[ServerSentEvent]:
    run_id = uuid4().hex
    seq = 0

    def event(payload: dict[str, Any]) -> ServerSentEvent:
        nonlocal seq
        seq += 1
        data = AgentEvent(run_id=run_id, seq=seq, **payload)
        return ServerSentEvent(id=f"{run_id}:{seq}", event=data.type, data=data)

    yield event({"type": "run.start", "status": "running"})
    stream = agent.astream(request.question, _to_langchain_messages(request.history))
    try:
        async for payload in stream:
            yield event(payload)
    except Exception as error:
        logger.exception("streaming chat request failed")
        timed_out = isinstance(error, (APITimeoutError, TimeoutError))
        yield event(
            {
                "type": "error",
                "status": "error",
                "data": {
                    "code": "timeout" if timed_out else "agent_error",
                    "message": (
                        "回答等待超时，请稍后重试。"
                        if timed_out
                        else "Agent 暂时无法回答，请稍后重试。"
                    ),
                },
            }
        )
    else:
        yield event({"type": "run.end", "status": "success"})
    finally:
        with anyio.CancelScope(shield=True):
            await stream.aclose()


__all__ = [
    "ChatHistoryMessage",
    "ChatRequest",
    "get_document_agent",
    "router",
]
