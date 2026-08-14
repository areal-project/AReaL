import json
from typing import Optional

from src.typings import Role, ChatHistoryItemDict, ChatHistoryItem
from abc import ABC, abstractmethod


class ChatHistoryItemFactoryInterface(ABC):
    @abstractmethod
    def construct(
        self, chat_history_item_index: int, expected_role: Optional[Role] = None
    ) -> ChatHistoryItem:
        pass

    @abstractmethod
    def get_chat_history_item_dict_deep_copy(self) -> ChatHistoryItemDict:
        pass

    @abstractmethod
    def set(self, prompt_index: int, role: Role, content: str) -> None:
        pass


class ChatHistoryItemFactory(ChatHistoryItemFactoryInterface):
    def __init__(self, chat_history_item_dict_path: str):
        super().__init__()
        self._chat_history_item_dict = ChatHistoryItemDict.model_validate(
            json.load(open(chat_history_item_dict_path))
        )

    def construct(
        self, chat_history_item_index: int, expected_role: Optional[Role] = None
    ) -> ChatHistoryItem:
        result = self._chat_history_item_dict.value[str(chat_history_item_index)]
        if expected_role is not None:
            assert result.role == expected_role
        return result

    def get_chat_history_item_dict_deep_copy(self) -> ChatHistoryItemDict:
        return self._chat_history_item_dict.model_copy(deep=True)

    def set(self, prompt_index: int, role: Role, content: str) -> None:
        self._chat_history_item_dict.set_chat_history_item(prompt_index, role, content)
