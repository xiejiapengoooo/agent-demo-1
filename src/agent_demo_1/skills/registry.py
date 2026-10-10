import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, TextIO

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from yaml import safe_load

from ..events import execution_step

SkillText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RunSkillScriptInput(BaseModel):
    path: str = Field(min_length=1, description="相对于当前技能目录的 scripts/ 下路径")
    args: list[str] = Field(
        default_factory=list, description="脚本的命令行参数，不包含 python 和脚本路径"
    )


class ReadSkillResourceInput(BaseModel):
    path: str = Field(min_length=1, description="相对于当前技能目录的路径")


class SkillMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: SkillText
    description: SkillText
    path: Path = Field(exclude=True)


def _read_frontmatter(source: TextIO) -> dict[str, Any]:
    if source.readline().strip() != "---":
        raise ValueError("SKILL.md must begin with YAML frontmatter")
    lines = []
    for line in source:
        if line.strip() == "---":
            metadata = safe_load("".join(lines))
            if not isinstance(metadata, dict):
                raise ValueError("skill frontmatter must be a mapping")
            return metadata
        lines.append(line)
    raise ValueError("skill frontmatter is missing its closing delimiter")


class SkillRegistry:
    def __init__(self, directory: Path | None = None) -> None:
        skill_directory = directory if directory is not None else Path(__file__).parent
        self._skills: dict[str, SkillMetadata] = {}
        for path in sorted(skill_directory.glob("*/SKILL.md")):
            with path.open(encoding="utf-8") as source:
                values = _read_frontmatter(source)
                metadata = SkillMetadata.model_validate({**values, "path": path})
            if metadata.name in self._skills:
                raise ValueError(f"duplicate skill name: {metadata.name}")
            self._skills[metadata.name] = metadata

    def catalog(self) -> list[SkillMetadata]:
        """Expose names and descriptions without reading skill bodies."""
        return list(self._skills.values())

    def get(self, name: str) -> SkillMetadata:
        try:
            return self._skills[name]
        except KeyError as error:
            raise ValueError(f"unknown skill: {name}") from error

    def load(self, name: str) -> str:
        """Read instructions only after a skill has been selected."""
        with self.get(name).path.open(encoding="utf-8") as source:
            _read_frontmatter(source)
            instructions = source.read().strip()
        if not instructions:
            raise ValueError(f"skill has no instructions: {name}")
        return instructions

    def tools(self, name: str) -> list[BaseTool]:
        metadata = self.get(name)
        tools = []
        if (metadata.path.parent / "scripts").is_dir():
            tools.append(self._make_script_tool(metadata.path.parent))
        if any(
            (metadata.path.parent / directory).is_dir()
            for directory in ("references", "assets", "scripts")
        ):
            tools.append(self._make_resource_tool(metadata.path.parent))
        return tools

    def _make_script_tool(self, skill_directory: Path) -> BaseTool:
        results: dict[str, str | None] = {}

        @tool(
            "run_skill_script",
            args_schema=RunSkillScriptInput,
            response_format="content_and_artifact",
        )
        def run_skill_script(path: str, args: list[str] | None = None) -> str:
            script = (skill_directory / path).resolve()
            if not script.is_relative_to(skill_directory):
                raise ValueError(
                    "script path escapes the selected skill's scripts directory"
                )
            if script.suffix != ".py" or not script.is_file():
                raise ValueError("skill script must be an existing Python file")
            arguments = args or []
            key = " ".join((str(script), *arguments))
            with execution_step("run_skill_script", tool=True, script=path):
                if key not in results:
                    process = subprocess.run(
                        [sys.executable, str(script), *arguments],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        timeout=10,
                        check=True,
                    )
                    results[key] = process.stdout.strip() or None
            return json.dumps(results, ensure_ascii=False)

        return run_skill_script

    def _make_resource_tool(self, skill_directory: Path) -> BaseTool:
        @tool(
            "read_skill_resource",
            args_schema=RunSkillScriptInput,
            response_format="content_and_artifact",
        )
        def read_skill_resource(path: str) -> str:
            resource = (skill_directory / path).resolve()
            if not resource.is_relative_to(skill_directory):
                raise ValueError("resource path escapes the selected skill directory")
            with execution_step("read_skill_resource", tool=True):
                return resource.read_text(encoding="utf-8")

        return read_skill_resource


__all__ = ["SkillMetadata", "SkillRegistry"]
