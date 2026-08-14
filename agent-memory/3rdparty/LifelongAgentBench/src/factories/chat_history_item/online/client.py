from typing import Optional

from src.factories.chat_history_item.online.chat_history_item_factory import (
    ChatHistoryItemFactoryInterface,
)
from src.utils import Client
from src.typings import (
    Role,
    ChatHistoryItem,
    ChatHistoryItemFactoryRequest,
    ChatHistoryItemFactoryResponse,
    ChatHistoryItemDict,
)


class ChatHistoryItemFactoryClient(Client, ChatHistoryItemFactoryInterface):
    def __init__(self, server_address: str, request_timeout: int):
        Client.__init__(
            self, server_address=server_address, request_timeout=request_timeout
        )

    def construct(  # type: ignore[override]  # noqa
        self, chat_history_item_index: int, expected_role: Optional[Role] = None
    ) -> ChatHistoryItem:
        response: ChatHistoryItemFactoryResponse.Construct = self._call_server(
            "/construct",
            ChatHistoryItemFactoryRequest.Construct(
                chat_history_item_index=chat_history_item_index,
                expected_role=expected_role,
            ),
            ChatHistoryItemFactoryResponse.Construct,
        )
        return response.chat_history_item

    def get_chat_history_item_dict_deep_copy(self) -> ChatHistoryItemDict:
        response: ChatHistoryItemFactoryResponse.GetChatHistoryItemDictDeepCopy = (
            self._call_server(
                "/get_chat_history_item_dict_deep_copy",
                None,
                ChatHistoryItemFactoryResponse.GetChatHistoryItemDictDeepCopy,
            )
        )
        return response.chat_history_item_dict

    def set(self, prompt_index: int, role: Role, content: str) -> None:
        _ = self._call_server(
            "/set",
            ChatHistoryItemFactoryRequest.Set(
                prompt_index=prompt_index, role=role, content=content
            ),
            None,
        )
