from typing import Any, Optional, Mapping
from typing_extensions import override

from src.agents.agent import Agent
from src.typings import (
    ChatHistoryItem,
    ChatHistory,
    LanguageModelContextLimitException,
    AgentContextLimitException,
    LanguageModelOutOfMemoryException,
    AgentOutOfMemoryException,
    LanguageModelUnknownException,
    AgentUnknownException,
    Role,
)
from src.language_models import LanguageModel


class LanguageModelAgent(Agent):
    def __init__(
        self,
        language_model: LanguageModel,
        system_prompt: str = "You are a helpful assistant.",
        inference_config_dict: Optional[Mapping[str, Any]] = None,
    ):
        """
        The name of the parameter `language_model` is referenced in `src.run_experiment.py` by string.
            So do not change it.
        """
        self._language_model = language_model
        self._system_prompt = system_prompt
        self._inference_config_dict = inference_config_dict

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        try:
            return self._language_model.inference(
                [chat_history], self._inference_config_dict, self._system_prompt
            )[0]
        except LanguageModelContextLimitException as e:
            raise AgentContextLimitException(str(e)) from e
        except LanguageModelOutOfMemoryException as e:
            raise AgentOutOfMemoryException(str(e)) from e
        except LanguageModelUnknownException as e:
            raise AgentUnknownException(str(e)) from e

    @override
    def get_role_dict(self) -> Mapping[Role, str]:
        return self._language_model.role_dict
