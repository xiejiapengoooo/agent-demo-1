from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any

from .base import (
    BaseParser,
    ParserSource,
    normalize_extension,
    normalize_mime_type,
)

ParserClass = type[BaseParser[Any]]


class _ParserRegistryMeta(type):
    """Bind public registry operations to the default instance on the class.

    This supports both ``ParserRegistry.register(...)`` for small scripts and
    ``ParserRegistry().register(...)`` when isolation is useful.  The actual
    implementation remains instance-based in both cases.
    """

    _class_dispatch = frozenset(
        {
            "register",
            "unregister",
            "find",
            "resolve",
            "get_parser_class",
            "get_parser",
            "create",
            "parse",
            "parsers",
            "extensions",
            "mime_types",
        }
    )

    def __getattribute__(cls, name: str) -> Any:
        attribute = super().__getattribute__(name)
        if name not in _ParserRegistryMeta._class_dispatch:
            return attribute
        default = cls.__dict__.get("_default_instance")
        if default is None:
            return attribute
        return attribute.__get__(default, cls)


class ParserRegistrationError(ValueError):
    """Raised when a parser cannot be registered safely."""


class ParserNotFoundError(LookupError):
    """Raised when no parser is registered for a source."""


def _as_values(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


class ParserRegistry(metaclass=_ParserRegistryMeta):
    def __init__(self) -> None:
        self._by_extension: dict[str, ParserClass] = {}
        self._by_mime_type: dict[str, ParserClass] = {}
        self._parser_order: list[ParserClass] = []

    @staticmethod
    def _is_parser_class(value: object) -> bool:
        return isinstance(value, type) and issubclass(value, BaseParser)

    def register(
        self,
        parser: ParserClass | str,
        *extensions: str | ParserClass,
        mime_types: str | Iterable[str] | None = None,
        override: bool = False,
        replace: bool | None = None,
    ) -> ParserClass:
        if replace is not None:
            override = replace

        if (
            isinstance(parser, str)
            and extensions
            and self._is_parser_class(extensions[0])
        ):
            parser_class = extensions[0]
            explicit_extensions = (parser, *extensions[1:])
            return self._register_class(
                parser_class,
                explicit_extensions,
                mime_types=mime_types,
                override=override,
            )

        if isinstance(parser, str):
            if not extensions or not self._is_parser_class(extensions[0]):
                raise TypeError(
                    "explicit registration requires register(ParserClass, ...) "
                    "or register(extension, ParserClass, ...)"
                )
            parser_class = extensions[0]
            explicit_extensions = (parser, *extensions[1:])
        else:
            parser_class = parser
            explicit_extensions = extensions

        if not self._is_parser_class(parser_class):
            raise TypeError("parser must be a BaseParser subclass")
        if any(not isinstance(extension, str) for extension in explicit_extensions):
            raise TypeError("parser extensions must be strings")
        return self._register_class(
            parser_class,
            explicit_extensions,
            mime_types=mime_types,
            override=override,
        )

    def _register_class(
        self,
        parser: ParserClass,
        explicit_extensions: Iterable[str],
        *,
        mime_types: str | Iterable[str] | None,
        override: bool,
    ) -> ParserClass:
        extension_values = tuple(explicit_extensions)
        if not extension_values:
            extension_values = _as_values(parser.extensions)
        mime_values = _as_values(mime_types)
        if not mime_values:
            mime_values = _as_values(parser.mime_types)

        try:
            normalized_extensions = tuple(
                dict.fromkeys(normalize_extension(value) for value in extension_values)
            )
            normalized_mime_types = tuple(
                dict.fromkeys(normalize_mime_type(value) for value in mime_values)
            )
        except (TypeError, ValueError) as exc:
            raise ParserRegistrationError(str(exc)) from exc

        if not normalized_extensions and not normalized_mime_types:
            raise ParserRegistrationError(
                f"{parser.__name__} must declare at least one extension or MIME type"
            )

        collisions = [
            extension
            for extension in normalized_extensions
            if extension in self._by_extension
            and self._by_extension[extension] is not parser
        ]
        collisions.extend(
            mime_type
            for mime_type in normalized_mime_types
            if mime_type in self._by_mime_type
            and self._by_mime_type[mime_type] is not parser
        )
        if collisions and not override:
            joined = ", ".join(sorted(collisions))
            raise ParserRegistrationError(f"parser already registered for: {joined}")

        for extension in normalized_extensions:
            self._by_extension[extension] = parser
        for mime_type in normalized_mime_types:
            self._by_mime_type[mime_type] = parser
        if parser not in self._parser_order:
            self._parser_order.append(parser)
        return parser

    def unregister(self, parser_or_key: ParserClass | str) -> None:
        """Remove a parser class or one extension/MIME registration."""

        if self._is_parser_class(parser_or_key):
            parser = parser_or_key
            for mapping in (self._by_extension, self._by_mime_type):
                for key, value in tuple(mapping.items()):
                    if value is parser:
                        del mapping[key]
            if parser in self._parser_order:
                self._parser_order.remove(parser)
            return

        if not isinstance(parser_or_key, str):
            raise TypeError(
                "parser or extension/MIME type must be a string or parser class"
            )
        key = parser_or_key.strip().lower()
        if "/" in key:
            key = normalize_mime_type(key)
            self._by_mime_type.pop(key, None)
        else:
            key = normalize_extension(key)
            self._by_extension.pop(key, None)

    def find(
        self,
        source: ParserSource | str,
        *,
        mime_type: str | None = None,
    ) -> ParserClass | None:
        """Find a parser class for a path, extension, or MIME type."""

        extension = self._source_extension(source)
        if extension is not None and extension in self._by_extension:
            return self._by_extension[extension]

        if mime_type is not None:
            try:
                normalized_mime = normalize_mime_type(mime_type)
            except (TypeError, ValueError):
                return None
            return self._by_mime_type.get(normalized_mime)

        if isinstance(source, str) and "/" in source:
            try:
                return self._by_mime_type.get(normalize_mime_type(source))
            except (TypeError, ValueError):
                return None
        return None

    resolve = find

    def get_parser_class(
        self,
        source: ParserSource | str,
        *,
        mime_type: str | None = None,
    ) -> ParserClass:
        """Return the registered parser class or raise a descriptive error."""

        parser = self.find(source, mime_type=mime_type)
        if parser is None:
            raise ParserNotFoundError(f"no parser registered for {source!r}")
        return parser

    def get_parser(
        self,
        source: ParserSource | str,
        *,
        mime_type: str | None = None,
        **kwargs: Any,
    ) -> BaseParser[Any]:
        """Create a parser instance selected for ``source``.

        ``source`` is passed to the parser constructor for path-like input.
        Extension and MIME lookup tokens (for example ``".pdf"``) create an
        unbound parser, which is useful when callers only need to inspect the
        selected implementation.
        """

        parser_class = self.get_parser_class(source, mime_type=mime_type)
        constructor_source: ParserSource | None = (
            source if self._looks_like_path(source) else None
        )
        return parser_class(constructor_source, **kwargs)

    create = get_parser

    def parse(
        self,
        source: ParserSource,
        *,
        mime_type: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Select a parser and parse ``source`` in one call."""

        parser = self.get_parser(source, mime_type=mime_type)
        return parser.parse(source, **kwargs)

    def parsers(self) -> tuple[ParserClass, ...]:
        """Return registered parser classes in registration order."""

        return tuple(self._parser_order)

    def extensions(self) -> frozenset[str]:
        return frozenset(self._by_extension)

    def mime_types(self) -> frozenset[str]:
        return frozenset(self._by_mime_type)

    def __len__(self) -> int:
        return len(self._parser_order)

    def __contains__(self, source: object) -> bool:
        if not isinstance(source, str):
            return False
        return self.find(source) is not None

    @staticmethod
    def _looks_like_path(source: ParserSource | str) -> bool:
        if isinstance(source, (Path, PathLike)):
            return True
        if not isinstance(source, str):
            return False
        return "/" in source or "\\" in source or bool(Path(source).suffix)

    @staticmethod
    def _source_extension(source: ParserSource | str) -> str | None:
        if isinstance(source, (Path, PathLike)):
            suffix = Path(source).suffix
            return suffix.lower() or None
        if not isinstance(source, str):
            return None

        value = source.strip()
        if not value:
            return None
        if value.startswith(".") and "/" not in value and "\\" not in value:
            try:
                return normalize_extension(value)
            except (TypeError, ValueError):
                return None
        if "/" not in value and "\\" not in value and not Path(value).suffix:
            try:
                return normalize_extension(value)
            except (TypeError, ValueError):
                return None
        suffix = Path(value).suffix
        return suffix.lower() or None


default_registry = ParserRegistry()
ParserRegistry._default_instance = default_registry
registry = default_registry


__all__ = [
    "ParserNotFoundError",
    "ParserRegistrationError",
    "ParserRegistry",
    "default_registry",
    "registry",
]
