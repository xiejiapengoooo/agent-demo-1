from pathlib import Path

from .parsers import DocxParser, ParserRegistry


def run():
    parser_registry = ParserRegistry()
    parser_registry.register(DocxParser)

    file = Path(
        "source/迪士尼乐园酒店信息 包括各家迪士尼酒店的房型、定价、设施（泳池、健身房）、入住_退房政策和酒店宾客专属福利.docx"
    )

    parser_registry.parse(file)


__all__ = ["run"]
