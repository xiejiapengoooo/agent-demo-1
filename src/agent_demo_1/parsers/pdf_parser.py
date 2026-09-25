from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseParser, ParserSource
from .convert_mineru_content_list import convert_mineru_content_list_v2
from .providers import BaseParserProvider, MineruCliProvider


class PdfParser(BaseParser[Any]):
    name: ClassVar[str] = "pdf"
    extensions: ClassVar[tuple[str, ...]] = (".pdf",)
    mime_types: ClassVar[tuple[str, ...]] = ("application/pdf",)

    def __init__(
        self,
        source: ParserSource | None = None,
        *,
        provider: BaseParserProvider | None = None,
    ) -> None:
        super().__init__(source)
        self.mineru_provider = provider if provider is not None else MineruCliProvider()

    def parse(self, source: ParserSource | None = None, **kwargs: Any) -> Any:
        source = self.validate_source(source)
        result = self.mineru_provider.parse(source)
        return {
            "source": source,
            "output": convert_mineru_content_list_v2(result),
        }


__all__ = ["PdfParser"]
