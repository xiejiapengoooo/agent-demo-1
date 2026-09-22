from collections.abc import Mapping, Sequence
from typing import Any

from bs4 import BeautifulSoup

from .summary import image_source, image_summary

KNOWN_TYPES = frozenset({"image", "table", "text"})


def normalize_blocks(
    blocks: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    result = []

    for block in blocks:
        type = block.get("type")
        if type not in KNOWN_TYPES:
            raise ValueError(f"Unknown type: {type}")

        normalized = {
            "type": block.get("type"),
            "page_idx": block.get("page_idx"),
            "text": "",
        }

        if block.get("type") == "text":
            normalized["text"] = block.get("text", "")

        if block.get("type") == "image":
            img_path = image_source(block.get("img_path", ""))
            normalized["source"] = img_path

            summary = (
                image_summary(
                    img_path,
                    caption="\n".join(block.get("image_caption", [])),
                    footnote="\n".join(block.get("image_footnote", [])),
                ).strip()
                or ""
            )
            if summary:
                normalized["text"] = summary

        if block.get("type") == "table":
            normalized["text"] = (
                block.get("table_body", "")
                + "\n"
                + "\n".join(block.get("table_footnote", []))
                + "\n"
                + "\n".join(block.get("table_caption", []))
                + "\n"
            ).strip() or ""

        result.append(normalized)

    return result


__all__ = ["normalize_blocks"]
