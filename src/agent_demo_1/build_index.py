from pathlib import Path

from .parsers import DocxParser, ParserRegistry


def run():
    parser_registry = ParserRegistry()
    parser_registry.register(DocxParser)

    file = Path(
        "source/迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx"
    )

    print(parser_registry.parse(file))


__all__ = ["run"]
