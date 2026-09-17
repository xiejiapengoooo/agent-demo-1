import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from openai import OpenAI

from .parsers import DocxParser, ParserRegistry
from .summary import image_summary

CHUNK_SIZE = 500
_ISOLATED_CHUNK_TYPES = frozenset({"image", "table"})


def chunk_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = CHUNK_SIZE,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks: list[dict[str, Any]] = []
    text_blocks: list[str] = []

    def flush_text_blocks() -> None:
        if not text_blocks:
            return

        text = "\n".join(text_blocks)
        for start in range(0, len(text), chunk_size):
            chunks.append(
                {
                    "type": "text",
                    "text": text[start : start + chunk_size],
                }
            )
        text_blocks.clear()

    for block in blocks:
        block_type = block.get("type")
        if block_type in _ISOLATED_CHUNK_TYPES:
            flush_text_blocks()
            chunks.append(dict(block))
            continue

        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                text_blocks.append(text)
            continue

        # Preserve unsupported block types instead of silently dropping them.
        flush_text_blocks()
        chunks.append(dict(block))

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


def run(chunk_size: int = CHUNK_SIZE):
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
    chunks = chunk_blocks(output, chunk_size=chunk_size)
    print(json.dumps(chunks, ensure_ascii=False, indent=2))


__all__ = ["chunk_blocks", "run"]
