import re
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ISOLATED_CHUNK_TYPES = frozenset({"image", "table"})
SENTENCE_END_RE = re.compile(r"[。！？!?；;.](?:[”’」』】）》）]*)|\n+")


def chunk_blocks(
    blocks: Sequence[Mapping[str, Any]],
    document_id: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    text_blocks: list[tuple[str, Any]] = []

    def flush_text_blocks() -> None:
        if not text_blocks:
            return

        text_parts: list[str] = []
        page_spans: list[tuple[int, int, Any]] = []
        cursor = 0
        for text_block, page_idx in text_blocks:
            if text_parts:
                text_parts.append("\n")
                cursor += 1
            text_parts.append(text_block)
            page_spans.append((cursor, cursor + len(text_block), page_idx))
            cursor += len(text_block)

        text = "".join(text_parts)
        for start, end in _text_chunk_ranges(text, CHUNK_SIZE, CHUNK_OVERLAP):
            page_start, page_end = _chunk_page_range(page_spans, start, end)
            chunks.append(
                {
                    "type": "text",
                    "text": text[start:end],
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )
        text_blocks.clear()

    for block in blocks:
        block_type = block.get("type")
        if block_type in ISOLATED_CHUNK_TYPES:
            flush_text_blocks()
            isolated_chunk = dict(block)
            page_idx = isolated_chunk.get("page_idx")
            isolated_chunk.setdefault("page_start", page_idx)
            isolated_chunk.setdefault("page_end", page_idx)
            chunks.append(isolated_chunk)
            continue

        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                text_blocks.append((text, block.get("page_idx")))
            continue

        # Preserve unsupported block types instead of silently dropping them.
        flush_text_blocks()
        isolated_chunk = dict(block)
        page_idx = isolated_chunk.get("page_idx")
        isolated_chunk.setdefault("page_start", page_idx)
        isolated_chunk.setdefault("page_end", page_idx)
        chunks.append(isolated_chunk)

    flush_text_blocks()

    for chunk_index, chunk in enumerate(chunks):
        chunk["order"] = chunk_index

    return chunks


def _text_chunk_ranges(text: str, chunk_size: int, chunk_overlap: int):
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end

        if hard_end < len(text):
            # Avoid producing a very short chunk merely because an early
            # sentence boundary happens to exist within the size window. The
            # boundary must also leave room to advance after applying overlap.
            minimum_end = start + max(chunk_overlap + 1, chunk_size // 2)
            sentence_ends = (
                match.end()
                for match in SENTENCE_END_RE.finditer(text, start, hard_end)
                if match.end() >= minimum_end
            )
            end = max(sentence_ends, default=hard_end)

            # Keep whitespace following the sentence ending with that chunk.
            while end < hard_end and text[end].isspace():
                end += 1

        yield start, end
        if end == len(text):
            break
        start = end - chunk_overlap


def _chunk_page_range(
    page_spans: Sequence[tuple[int, int, Any]],
    start: int,
    end: int,
) -> tuple[Any, Any]:
    pages = [
        page
        for span_start, span_end, page in page_spans
        if page is not None and span_start < end and span_end > start
    ]
    if not pages:
        return None, None
    return pages[0], pages[-1]


__all__ = ["chunk_blocks"]
