from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage


class BaseAgent(ABC):
    name: ClassVar[str]
    system_prompt: ClassVar[str]

    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def _with_system_prompt(
        self,
        messages: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        return [SystemMessage(content=self.system_prompt), *messages]

    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Run the agent for one orchestration step."""


__all__ = ["BaseAgent"]
