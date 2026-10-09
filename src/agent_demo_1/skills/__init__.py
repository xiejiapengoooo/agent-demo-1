from pathlib import Path

from .chat_export import ExportedFile, export_chat_history

CHAT_EXPORT_INSTRUCTIONS = (
    Path(__file__).parent / "chat-export" / "SKILL.md"
).read_text(encoding="utf-8")

__all__ = ["CHAT_EXPORT_INSTRUCTIONS", "ExportedFile", "export_chat_history"]
