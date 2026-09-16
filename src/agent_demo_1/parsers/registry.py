from __future__ import annotations

from typing import Any

from .base import (
    BaseParser,
    ParserSource,
)

ParserClass = type[BaseParser[Any]]


class ParserRegistrationError(ValueError):
    """Raised when a parser cannot be registered safely."""


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: list[ParserClass] = []

    @staticmethod
    def _is_parser_class(value: object) -> bool:
        return isinstance(value, type) and issubclass(value, BaseParser)

    def register(self, parser: ParserClass) -> ParserClass:
        if not self._is_parser_class(parser):
            raise TypeError("parser must be a BaseParser subclass")
        return self._register_class(parser)

    def _register_class(self, parser: ParserClass) -> ParserClass:
        if parser in self._parsers:
            return parser

        try:
            normalized_extensions = parser.normalized_extensions()
            normalized_mime_types = parser.normalized_mime_types()
        except (TypeError, ValueError) as exc:
            raise ParserRegistrationError(str(exc)) from exc

        if not normalized_extensions and not normalized_mime_types:
            raise ParserRegistrationError(
                f"{parser.name} must declare at least one extension or MIME type"
            )

        registered_extensions = self.extensions()
        registered_mime_types = self.mime_types()
        collisions = normalized_extensions & registered_extensions
        collisions |= normalized_mime_types & registered_mime_types
        if collisions:
            joined = ", ".join(sorted(collisions))
            raise ParserRegistrationError(f"parser already registered for: {joined}")

        self._parsers.append(parser)
        return parser

    def unregister(self, parser: ParserClass) -> None:
        if not self._is_parser_class(parser):
            raise TypeError("parser must be a BaseParser subclass")

        if parser in self._parsers:
            self._parsers.remove(parser)

    def get_parser_class(
        self,
        source: ParserSource | str,
        *,
        mime_type: str | None = None,
    ) -> ParserClass:
        for parser in self._parsers:
            if parser.can_parse(source, mime_type=mime_type):
                return parser

        raise LookupError(f"no parser registered for {source!r}")

    def get_parser(
        self,
        source: ParserSource | str,
        *,
        mime_type: str | None = None,
        **kwargs: Any,
    ) -> BaseParser[Any]:
        parser_class = self.get_parser_class(source, mime_type=mime_type)
        return parser_class(source, **kwargs)

    def parse(
        self,
        source: ParserSource,
        *,
        mime_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        parser = self.get_parser(source, mime_type=mime_type)
        return parser.parse(source, **kwargs)

    def parsers(self) -> tuple[ParserClass, ...]:
        return tuple(self._parsers)

    def extensions(self) -> frozenset[str]:
        return frozenset(
            extension
            for parser in self._parsers
            for extension in parser.normalized_extensions()
        )

    def mime_types(self) -> frozenset[str]:
        return frozenset(
            mime_type
            for parser in self._parsers
            for mime_type in parser.normalized_mime_types()
        )


__all__ = [
    "ParserRegistrationError",
    "ParserRegistry",
]
