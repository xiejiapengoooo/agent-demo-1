from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

OUTPUT_DIRECTORY = Path("static") / "output"


class ExportedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_name: str
    url: str


def export_chat_history(history: Sequence[BaseMessage]) -> ExportedFile | None:
    """Export the supplied conversation without asking the model to rewrite it."""
    sections = []
    for message in history:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            continue
        content = message.content if isinstance(message.content, str) else message.text
        sections.append(f"## {len(sections) + 1}. {role}\n\n{content}")

    if not sections:
        return None

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_name = f"chat-{uuid4().hex}.md"
    markdown = "# 聊天记录\n\n" + "\n\n---\n\n".join(sections) + "\n"
    with (OUTPUT_DIRECTORY / file_name).open(
        "x", encoding="utf-8", newline="\n"
    ) as output:
        output.write(markdown)
    return ExportedFile(file_name=file_name, url=f"/output/{file_name}")


__all__ = ["ExportedFile", "export_chat_history"]
