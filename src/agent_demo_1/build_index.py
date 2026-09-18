import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

from .embedding import embed_chunks


def build_index():
    # parser_registry = ParserRegistry()
    # parser_registry.register(DocxParser)

    # file = Path(
    #     "source/迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx"
    # )

    # result = parser_registry.parse(file)
    # if not result or not result.get("output"):
    #     raise ValueError("no output")

    # data = result.get("output")

    # blocks = normalize_blocks(data)

    # chunks = chunk_blocks(
    #     blocks,
    #     document_id="迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx",
    # )
    #
    # embedded_chunks = embed_chunks(chunks)

    with Path("mock_1.json").open(encoding="utf-8") as data_file:
        embedded_chunks = json.load(data_file)

    print(embedded_chunks)


__all__ = ["build_index"]
