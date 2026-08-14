from abc import ABC, abstractmethod
from typing import Sequence, Mapping, Any, Optional, final

from src.typings import (
    ChatHistory,
    ChatHistoryItem,
    Role,
    ModelException,
    LanguageModelUnknownException,
)


class LanguageModel(ABC):
    def __init__(self, role_dict: Mapping[str, str]) -> None:
        self.role_dict: Mapping[Role, str] = {
            Role(role): role_dict[role] for role in Role
        }

    def _convert_chat_history_to_message_list(
        self, chat_history: ChatHistory
    ) -> list[Mapping[str, str]]:
        message_list: list[Mapping[str, str]] = []
        for item_index in range(chat_history.get_value_length()):
            chat_history_item = chat_history.get_item_deep_copy(item_index)
            message_list.append(
                {
                    "role": self.role_dict[chat_history_item.role],
                    "content": chat_history_item.content,
                }
            )
        return message_list

    @final
    def inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Optional[Mapping[str, Any]] = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Sequence[ChatHistoryItem]:
        for chat_history in batch_chat_history:
            assert chat_history.get_item_deep_copy(-1).role == Role.USER
        try:
            if inference_config_dict is None:
                inference_config_dict = {}
            inference_result = self._inference(
                batch_chat_history, inference_config_dict, system_prompt
            )
        except ModelException as e:
            raise e
        except Exception as e:
            raise LanguageModelUnknownException(str(e)) from e
        return inference_result

    @abstractmethod
    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        pass
