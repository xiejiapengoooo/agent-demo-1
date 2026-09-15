from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

ParserSource = str | PathLike[str] | Path


def normalize_extension(extension: str) -> str:
    if not isinstance(extension, str):
        raise TypeError("file extension must be a string")

    value = extension.strip().lower()
    if not value:
        raise ValueError("file extension cannot be empty")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"invalid file extension: {extension!r}")

    value = value.lstrip(".")
    if not value or "." in value:
        raise ValueError(f"invalid file extension: {extension!r}")
    return f".{value}"


def normalize_mime_type(mime_type: str) -> str:
    if not isinstance(mime_type, str):
        raise TypeError("MIME type must be a string")

    value = mime_type.split(";", 1)[0].strip().lower()
    if not value or "/" not in value or value.startswith("/") or value.endswith("/"):
        raise ValueError(f"invalid MIME type: {mime_type!r}")
    return value


class BaseParser[ParserResultT](ABC):
    name: ClassVar[str]
    extensions: ClassVar[tuple[str, ...] | frozenset[str] | str] = ()
    mime_types: ClassVar[tuple[str, ...] | frozenset[str] | str] = ()

    def __init__(self, source: ParserSource | None = None) -> None:
        self.source = self._coerce_source(source) if source is not None else None

    @staticmethod
    def _coerce_source(source: ParserSource) -> Path:
        if isinstance(source, (str, PathLike)):
            return Path(source)
        raise TypeError("parser source must be a path-like value")

    @classmethod
    def normalized_extensions(cls) -> frozenset[str]:
        values = cls.extensions
        if isinstance(values, str):
            values = (values,)
        return frozenset(normalize_extension(value) for value in values)

    @classmethod
    def normalized_mime_types(cls) -> frozenset[str]:
        values = cls.mime_types
        if isinstance(values, str):
            values = (values,)
        return frozenset(normalize_mime_type(value) for value in values)

    @classmethod
    def can_parse(
        cls,
        source: ParserSource | str,
        *,
        mime_type: str | None = None,
    ) -> bool:
        extension: str | None = None
        if isinstance(source, (Path, PathLike)):
            extension = Path(source).suffix.lower() or None
        elif isinstance(source, str):
            candidate = source.strip()
            if "/" in candidate and "\\" not in candidate:
                try:
                    if normalize_mime_type(candidate) in cls.normalized_mime_types():
                        return True
                except (TypeError, ValueError):
                    pass
            if "/" not in candidate and "\\" not in candidate:
                suffix = Path(candidate).suffix
                if suffix:
                    extension = suffix.lower()
                else:
                    try:
                        extension = normalize_extension(candidate)
                    except (TypeError, ValueError):
                        return False
            else:
                extension = Path(candidate).suffix.lower() or None

        extension_match = (
            extension is not None and extension in cls.normalized_extensions()
        )
        mime_match = False
        if mime_type is not None:
            try:
                mime_match = (
                    normalize_mime_type(mime_type) in cls.normalized_mime_types()
                )
            except (TypeError, ValueError):
                return False

        return extension_match or mime_match

    def validate_source(self, source: ParserSource | None = None) -> Path:
        if source is not None:
            self.source = self._coerce_source(source)
        if self.source is None:
            raise ValueError("a parser source is required")
        if not self.source.exists():
            raise FileNotFoundError(self.source)
        if not self.source.is_file():
            raise IsADirectoryError(self.source)
        return self.source

    @abstractmethod
    def parse(self, source: ParserSource | None = None, **kwargs: Any) -> ParserResultT:
        """Parse ``source`` and return the parser-specific result."""


__all__ = [
    "BaseParser",
    "ParserSource",
    "normalize_extension",
    "normalize_mime_type",
]
