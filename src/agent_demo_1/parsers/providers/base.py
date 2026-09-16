from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ..base import ParserSource


@dataclass(frozen=True, slots=True)
class ParserProviderResult:
    source: Path
    output: Path


class BaseParserProvider(ABC):
    name: ClassVar[str]

    @abstractmethod
    def parse(
        self,
        source: ParserSource,
    ) -> ParserProviderResult:
        """Parse ``source`` and write provider artifacts."""


__all__ = ["BaseParserProvider", "ParserProviderResult"]
