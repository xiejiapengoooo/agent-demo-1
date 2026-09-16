from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseParser, ParserSource
from .providers import (
    BaseParserProvider,
    MineruCliProvider,
    ParserProviderResult,
)


class DocxParser(BaseParser[ParserProviderResult]):
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

    def parse(
        self, source: ParserSource | None = None, **kwargs: Any
    ) -> ParserProviderResult:
        return self.mineru_provider.parse(self.validate_source(source))


__all__ = ["DocxParser"]
