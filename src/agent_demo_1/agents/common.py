from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str
    chunk_id: str
    file_name: str
    page_start: int | None = None
    page_end: int | None = None
    score: float = Field(ge=-1, le=1)


class Evidence(Source):
    text: str = Field(min_length=1)


class AgentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[Source]
    route: Literal["direct", "research"]


def merge_evidence(
    current: list[Evidence] | None,
    new: list[Evidence] | None,
) -> list[Evidence]:
    merged: dict[str, Evidence] = {}
    for item in [*(current or []), *(new or [])]:
        previous = merged.get(item.chunk_id)
        if previous is None or item.score > previous.score:
            merged[item.chunk_id] = item
    return list(merged.values())


def normalize_chunks(chunks: Sequence[Mapping[str, Any]]) -> list[Evidence]:
    evidence = []
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise TypeError("retrieved chunk must have a non-empty id")

        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TypeError(f"retrieved chunk {chunk_id} must have non-empty text")

        file_name = chunk.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise TypeError(f"retrieved chunk {chunk_id} must have a file name")

        score = chunk.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise TypeError(f"retrieved chunk {chunk_id} must have a finite score")

        evidence.append(
            Evidence(
                citation_id=_citation_id(chunk_id),
                chunk_id=chunk_id,
                file_name=" ".join(file_name.split()),
                page_start=_display_page(chunk.get("page_start")),
                page_end=_display_page(chunk.get("page_end")),
                score=float(score),
                text=text.strip(),
            )
        )
    return merge_evidence([], evidence)


def format_evidence(evidence: Sequence[Evidence]) -> str:
    if not evidence:
        return "知识库中没有检索到相关文档片段。"

    sections = []
    for item in evidence:
        sections.append(
            "\n".join(
                (
                    f"引用：[{item.citation_id}]",
                    f"文件：{item.file_name}",
                    f"页码：{_page_label(item)}",
                    f"相关度：{item.score:.4f}",
                    "<document>",
                    item.text,
                    "</document>",
                )
            )
        )
    return "\n\n".join(sections)


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def last_user_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            value = message_text(message)
            if value:
                return value
    raise ValueError("conversation has no non-empty user message")


def extract_evidence(messages: Sequence[BaseMessage]) -> list[Evidence]:
    evidence = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        artifact = message.artifact
        if not isinstance(artifact, Sequence) or isinstance(artifact, (str, bytes)):
            continue
        for item in artifact:
            evidence.append(Evidence.model_validate(item))
    return merge_evidence([], evidence)


def _display_page(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"invalid page index: {value!r}")
    return value + 1


def _citation_id(chunk_id: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]", "", chunk_id)[:8]
    if not token:
        raise ValueError("chunk id cannot produce a citation id")
    return f"S-{token.lower()}"


def _page_label(item: Evidence) -> str:
    if item.page_start is None:
        return "未知"
    if item.page_end is None or item.page_end == item.page_start:
        return str(item.page_start)
    return f"{item.page_start}-{item.page_end}"


__all__ = [
    "AgentResult",
    "Evidence",
    "Source",
    "extract_evidence",
    "format_evidence",
    "last_user_text",
    "merge_evidence",
    "message_text",
    "normalize_chunks",
]
