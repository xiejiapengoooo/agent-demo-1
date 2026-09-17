import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from openai import OpenAI

from .chunking import chunk_blocks
from .normalizer import normalize_blocks
from .parsers import DocxParser, ParserRegistry


def build_index():
    # parser_registry = ParserRegistry()
    # parser_registry.register(DocxParser)

    # file = Path(
    #     "source/迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx"
    # )

    # result = parser_registry.parse(file)
    # if not result or not result.get("output"):
    #     raise ValueError("no output")

    # output = result.get("output")

    with Path("mock.json").open(encoding="utf-8") as data_file:
        data = json.load(data_file)

    blocks = normalize_blocks(data)

    chunks = chunk_blocks(blocks)

    print(json.dumps(chunks, ensure_ascii=False))


__all__ = ["build_index"]
