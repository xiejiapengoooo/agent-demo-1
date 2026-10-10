"""将 json 格式的聊天记录导出成 markdown 文件格式。"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

OUTPUT_DIR = Path("output")


def export_chat_history(history: list[dict]) -> dict:
    if not isinstance(history, list):
        raise TypeError("history must be a JSON array")
    sections = []
    for message in history:
        if not isinstance(message, dict):
            raise TypeError("each history message must be a JSON object")
        role = {"user": "用户", "assistant": "助手"}.get(message.get("role", ""))
        if role is None:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("each user or assistant message must have string content")
        sections.append(f"## {len(sections) + 1}. {role}\n\n{content}")

    if not sections:
        return {"files": []}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"chat-{uuid4().hex}.md"
    markdown = "# 聊天记录\n\n" + "\n\n---\n\n".join(sections) + "\n"
    with (OUTPUT_DIR / file_name).open("x", encoding="utf-8", newline="\n") as output:
        output.write(markdown)
    return {
        "files": [{"file_name": file_name, "url": f"/output/{file_name}"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="聊天记录 JSON 数组")
    arguments = parser.parse_args()
    try:
        export_chat_history(json.load(arguments.input))
    except (OSError, ValueError) as error:
        parser.exit(1, f"导出失败：{error}\n")


if __name__ == "__main__":
    main()
