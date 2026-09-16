from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseParser, ParserSource
from .convert_mineru_content_list import convert_mineru_content_list_v2
from .providers import (
    BaseParserProvider,
    MineruCliProvider,
)


class DocxParser(BaseParser[Any]):
    name: ClassVar[str] = "docx"
    extensions: ClassVar[tuple[str, ...]] = (".docx",)
    mime_types: ClassVar[tuple[str, ...]] = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    def __init__(
        self,
        source: ParserSource | None = None,
        *,
        provider: BaseParserProvider | None = None,
    ) -> None:
        super().__init__(source)
        self.mineru_provider = MineruCliProvider()

    def parse(self, source: ParserSource | None = None, **kwargs: Any) -> Any:
        result = self.mineru_provider.parse(self.validate_source(source))
        return convert_mineru_content_list_v2(result)


__all__ = ["DocxParser"]
