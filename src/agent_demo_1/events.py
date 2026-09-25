from collections.abc import Generator
from contextlib import contextmanager
from time import monotonic
from typing import Any, Literal
from uuid import uuid4

from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    run_id: str
    seq: int
    type: Literal[
        "run.start",
        "step.start",
        "step.end",
        "tool.start",
        "tool.end",
        "answer.delta",
        "answer.final",
        "run.end",
        "error",
    ]
    node: str | None = None
    step_id: str | None = None
    status: Literal["running", "success", "error"] | None = None
    data: dict[str, Any] = Field(default_factory=dict)


@contextmanager
def execution_step(
    node: str,
    *,
    tool: bool = False,
    **details: Any,
) -> Generator[dict[str, Any]]:
    writer = get_stream_writer()
    step_id = uuid4().hex
    kind = "tool" if tool else "step"
    started = monotonic()
    writer(
        {
            "type": f"{kind}.start",
            "node": node,
            "step_id": step_id,
            "status": "running",
            "data": dict(details),
        }
    )
    try:
        yield details
    except Exception:
        writer(
            {
                "type": f"{kind}.end",
                "node": node,
                "step_id": step_id,
                "status": "error",
                "data": {
                    **details,
                    "duration_ms": round((monotonic() - started) * 1000),
                },
            }
        )
        raise
    else:
        writer(
            {
                "type": f"{kind}.end",
                "node": node,
                "step_id": step_id,
                "status": "success",
                "data": {
                    **details,
                    "duration_ms": round((monotonic() - started) * 1000),
                },
            }
        )


__all__ = ["AgentEvent", "execution_step"]
