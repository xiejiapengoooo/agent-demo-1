from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "agent-demo-1"
    host: str = "0.0.0.0"
    port: int = 3000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    agent_model: str = "gpt-5.6-sol"
    openai_base_url: str = "https://www.su8.codes/v1"
    agent_timeout: float = Field(default=60.0, gt=0)
    agent_max_research_rounds: int = Field(default=2, ge=1)
    agent_max_revision_rounds: int = Field(default=1, ge=0)
    agent_recursion_limit: int = Field(default=20, ge=4)


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
