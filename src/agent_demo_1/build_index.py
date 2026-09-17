import json
from pathlib import Path

from bs4 import BeautifulSoup
from openai import OpenAI

from .parsers import DocxParser, ParserRegistry
from .summary import image_summary


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


def run():
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
    print(output)


__all__ = ["run"]
