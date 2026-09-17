from __future__ import annotations

import base64
import mimetypes
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from .parsers.providers.mineru_cli import MineruCliProvider

SUMMARY_MODEL = "gpt-4.1-mini"


def image_summary(
    image_path: str | PathLike[str],
    *,
    client: OpenAI,
    captions: str = "",
    footnotes: str = "",
) -> str:
    prompt = f"""你是一位专业的图像分析专家。请提供详细、准确的描述。请详细分析这张图片：

    对图片的全面详细描述，遵循以下指导：
    - 描述整体构图和布局
    - 识别所有对象、人物、文字和视觉元素
    - 解释元素之间的关系
    - 注意颜色、光照和视觉风格
    - 描述展示的任何动作或活动
    - 如涉及图表、图解等，包含技术细节
    - 始终使用具体名称而非代词",

    附加信息：
    - 标注：{captions}
    - 脚注：{footnotes}

    请专注于提供准确、详细的视觉分析，以便于知识检索。"""
    image_url = _image_url(image_path)
    response = client.responses.create(
        model=SUMMARY_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "auto",
                    },
                ],
            }
        ],
    )

    summary = response.output_text.strip()
    if not summary:
        raise ValueError("OpenAI returned an empty image summary")
    return summary


def _image_url(image_path: str | PathLike[str]) -> str:
    if not isinstance(image_path, (str, PathLike)):
        raise TypeError("image path must be a path-like value")

    value = str(image_path).strip()
    if not value:
        raise ValueError("image path cannot be empty")

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value
    if value.startswith("data:image/"):
        return value

    path = Path(image_path)
    output_root = MineruCliProvider.output_dir / path.stem
    matches = list(output_root.rglob(path.name)) if output_root.exists() else []
    if not matches:
        raise FileNotFoundError(path)
    if len(matches) > 1:
        candidates = ", ".join(str(candidate) for candidate in matches)
        raise ValueError(f"image path is ambiguous; found: {candidates}")

    mime_type, _ = mimetypes.guess_type(matches[0].name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"unable to determine image type: {path}")

    encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def table_summary():
    pass


__all__ = ["image_summary", "table_summary"]
