from pathlib import Path

from openai import OpenAI

from .parsers import DocxParser, ParserRegistry
from .summary import image_summary


def run():
    openai_client = OpenAI(base_url="https://token.xiejiapeng.com/v1")

    parser_registry = ParserRegistry()
    parser_registry.register(DocxParser)

    file = Path(
        "source/迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx"
    )

    result = parser_registry.parse(file)
    if not result or not result.get("output"):
        raise ValueError("no output")

    output = result.get("output")
    for _, block in enumerate(output):
        if block.get("type") == "image":
            summary = image_summary(
                block.get("img_path"),
                client=openai_client,
                captions="\n".join(block.get("image_caption", [])),
                footnotes="\n".join(block.get("image_footnote", [])),
            )
            print(summary)


__all__ = ["run"]
