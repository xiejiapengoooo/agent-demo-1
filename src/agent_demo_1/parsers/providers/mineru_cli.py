from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..base import ParserSource
from .base import BaseParserProvider


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
    ) -> Any:
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

        content_list_path = self._content_list_path(source_path)
        try:
            with content_list_path.open(encoding="utf-8") as content_file:
                return json.load(content_file)
        except OSError as exc:
            raise MineruCliError(
                f"MinerU content list could not be read: {content_list_path}"
            ) from exc
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise MineruCliError(
                f"MinerU content list is not valid JSON: {content_list_path}"
            ) from exc

    def _content_list_path(self, source: Path) -> Path:
        output_root = self.output_dir / source.stem
        expected_name = f"{source.stem}_content_list_v2.json"

        matches = list(output_root.rglob(expected_name)) if output_root.exists() else []

        if not matches:
            raise MineruCliError(
                "MinerU CLI did not produce a content_list_v2 JSON file for "
                f"{source.name!r} under {output_root}"
            )

        if len(matches) > 1:
            raise MineruCliError(
                "MinerU CLI produced multiple content_list_v2 JSON files for "
                f"{source.name!r} under {output_root}"
            )

        return matches[0]

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
