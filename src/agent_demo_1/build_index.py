import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from openai import OpenAI

from .parsers import DocxParser, ParserRegistry
from .summary import image_summary

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
_ISOLATED_CHUNK_TYPES = frozenset({"image", "table"})
_SENTENCE_END_RE = re.compile(r"[。！？!?；;.](?:[”’」』】）》）]*)|\n+")


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
                for match in _SENTENCE_END_RE.finditer(text, start, hard_end)
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


def chunk_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and less than chunk_size")

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
        for start, end in _text_chunk_ranges(text, chunk_size, chunk_overlap):
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
        if block_type in _ISOLATED_CHUNK_TYPES:
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
    return chunks


def html_table_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table")
    if not table:
        return ""

    rows = []

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        row = [cell.get_text(" ", strip=True).replace("|", "\\|") for cell in cells]
        if row:
            rows.append(row)

    if not rows:
        return ""

    header = rows[0]

    result = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for row in rows[1:]:
        # 防止列数不一致
        row = row + [""] * (len(header) - len(row))
        result.append("| " + " | ".join(row[: len(header)]) + " |")

    return "\n".join(result)


def run(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    # openai_client = OpenAI(base_url="https://token.xiejiapeng.com/v1")

    # parser_registry = ParserRegistry()
    # parser_registry.register(DocxParser)

    # file = Path(
    #     "source/迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx"
    # )

    # result = parser_registry.parse(file)
    # if not result or not result.get("output"):
    #     raise ValueError("no output")

    # output = result.get("output")
    #
    with Path("src/agent_demo_1/data.json").open(encoding="utf-8") as data_file:
        output = json.load(data_file)
    for _, block in enumerate(output):
        # if block.get("type") == "image":
        #     summary = image_summary(
        #         block.get("img_path"),
        #         client=openai_client,
        #         captions="\n".join(block.get("image_caption", [])),
        #         footnotes="\n".join(block.get("image_footnote", [])),
        #     )
        #     if summary:
        #         block["summary"] = summary
        if block.get("type") == "table":
            table_body = block.get("table_body", "")
            if table_body:
                markdown = html_table_to_markdown(table_body)
                if markdown:
                    block["table_body_markdown"] = markdown
    chunks = chunk_blocks(
        output,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(json.dumps(chunks, ensure_ascii=False))


__all__ = ["chunk_blocks", "run"]
