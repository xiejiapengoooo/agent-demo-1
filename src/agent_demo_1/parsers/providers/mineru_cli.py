from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ..base import ParserSource
from .base import BaseParserProvider, ParserProviderResult


class MineruCliError(RuntimeError):
    """Raised when MinerU CLI cannot produce a usable parse result."""


@dataclass(frozen=True, slots=True)
class MineruCliDefaultOptions:
    backend: str = "hybrid-engine"
    method: str = "auto"
    effort: str = "high"
    language: str = "ch"
    formula: bool = True
    table: bool = True
    image_analysis: bool = True
    client_side_output_generation: bool = False


class MineruCliProvider(BaseParserProvider):
    name: ClassVar[str] = "mineru-cli"
    executable: ClassVar[str] = "mineru"
    output_dir: ClassVar[Path] = Path(".mineru_output")

    def __init__(
        self,
        *,
        options: MineruCliDefaultOptions | None = None,
        timeout: float | None = None,
    ) -> None:
        self.options = options or MineruCliDefaultOptions()
        self.timeout = timeout

    def parse(
        self,
        source: ParserSource,
    ) -> ParserProviderResult:
        source_path = Path(source).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not source_path.is_file():
            raise IsADirectoryError(source_path)

        source_path = source_path.resolve()
        command = self.build_command(
            source_path,
        )

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise MineruCliError(
                f"MinerU executable was not found: {self.executable!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MineruCliError(
                f"MinerU CLI timed out while parsing {source_path.name!r}"
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            message = f"MinerU CLI exited with status {completed.returncode}"
            if detail:
                message = f"{message}: {detail}"
            raise MineruCliError(message)

        return ParserProviderResult(
            source=source_path,
            output=self.output_dir / source_path.stem / self.options.method,
        )

    def build_command(
        self,
        source: Path,
    ) -> tuple[str, ...]:
        options = self.options
        command = [
            self.executable,
            "--path",
            str(source),
            "--output",
            str(self.output_dir),
            "--method",
            options.method,
            "--backend",
            options.backend,
            "--effort",
            options.effort,
            "--lang",
            options.language,
            "--formula",
            str(options.formula).lower(),
            "--table",
            str(options.table).lower(),
            "--image-analysis",
            str(options.image_analysis).lower(),
            "--client-side-output-generation",
            str(options.client_side_output_generation).lower(),
        ]
        return tuple(command)


__all__ = ["MineruCliDefaultOptions", "MineruCliError", "MineruCliProvider"]
