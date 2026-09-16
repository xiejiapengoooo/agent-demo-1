from collections.abc import Mapping
from enum import Enum
from typing import Any

MINERU_V2_LAYOUT_TYPES = frozenset(
    {
        "page_header",
        "page_footer",
        "page_number",
        "page_aside_text",
        "page_footnote",
    }
)

TEXT_TYPES = frozenset(
    {
        "title",
        "paragraph",
    }
)

LIST_TYPES = frozenset({"list", "index"})


class FinalItem(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class MineruContentListV2Error(ValueError):
    """Raised when a value cannot be interpreted as a MinerU v2 content list."""


def convert_mineru_content_list_v2(
    payload: Any,
):
    _validate_payload(payload)

    converted: list[dict[str, Any]] = []
    for page_idx, page in enumerate(payload):
        for block in page:
            item = _convert_block(
                block,
                page_idx,
            )
            if item is not None:
                converted.append(item)

    if not converted:
        raise MineruContentListV2Error(
            "MinerU content_list_v2 does not contain any supported content blocks"
        )

    return converted


def _validate_payload(payload: Any) -> None:
    if not isinstance(payload, list) or not payload:
        raise MineruContentListV2Error(
            "MinerU content_list_v2 must be a non-empty list of pages"
        )

    has_block = False
    for page_idx, page in enumerate(payload):
        if not isinstance(page, list):
            raise MineruContentListV2Error(
                f"MinerU content_list_v2 page {page_idx} must be a list"
            )
        for block_idx, block in enumerate(page):
            has_block = True
            if not isinstance(block, Mapping):
                raise MineruContentListV2Error(
                    "MinerU content_list_v2 block "
                    f"{page_idx}:{block_idx} must be an object"
                )

    if not has_block:
        raise MineruContentListV2Error(
            "MinerU content_list_v2 does not contain any content blocks"
        )


def _block_content(block: Mapping[str, Any]) -> dict[str, Any]:
    content = block.get("content", {})
    if isinstance(content, Mapping):
        return dict(content)
    else:
        return {}


def _convert_block(
    block: Mapping[str, Any],
    page_idx: int,
) -> dict[str, Any] | None:
    block_type = block.get("type", "").strip()
    if not block_type:
        return None

    if block_type in MINERU_V2_LAYOUT_TYPES:
        return None

    content = _block_content(block)

    if block_type in TEXT_TYPES:
        content_key = f"{block_type}_content"
        text = _text_value(content.get(content_key)).strip()
        if not text:
            return None

        item: dict[str, Any] = {"type": FinalItem.TEXT, "text": text}
        if block_type == "title":
            level = _positive_int(content.get("level", block.get("level")))
            if level is not None:
                item["text_level"] = level
        return _final_item(item, block, page_idx, block_type)


def _final_item(
    item: dict[str, Any],
    block: Mapping[str, Any],
    page_idx: int,
    block_type: str,
) -> dict[str, Any]:
    item["page_idx"] = page_idx
    item["_mineru_v2_type"] = block_type

    bbox = block.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        item["bbox"] = list(bbox)

    return item


def _text_value(value: Any) -> str:
    if isinstance(value, list):
        result = ""
        for part in value:
            if not part:
                continue
            if (
                result
                and not result[-1].isspace()
                and not part[0].isspace()
                and _needs_span_separator(result[-1], part[0])
            ):
                result += " "
            result += part
        return result
    return ""


def _needs_span_separator(left: str, right: str) -> bool:
    if left.isspace() or right.isspace():
        return False
    if _is_cjk(left) or _is_cjk(right):
        return False
    return not (right in ",.;:!?)]}" or left in "([{")


def _is_cjk(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff"


def _positive_int(value: Any) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
