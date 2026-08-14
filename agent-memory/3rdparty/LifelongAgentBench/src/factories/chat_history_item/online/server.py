from fastapi import FastAPI, APIRouter
import uvicorn

from .chat_history_item_factory import ChatHistoryItemFactory

from src.typings import (
    ChatHistoryItemFactoryRequest,
    ChatHistoryItemFactoryResponse,
)
from src.utils import Server


class ChatHistoryItemFactoryServer(Server):
    def __init__(
        self, router: APIRouter, chat_history_item_factory: ChatHistoryItemFactory
    ):
        Server.__init__(self, router, chat_history_item_factory)
        self._chat_history_item_factory = chat_history_item_factory
        self.router.post("/construct")(self.construct)
        self.router.post("/get_chat_history_item_dict_deep_copy")(
            self.get_chat_history_item_dict_deep_copy
        )
        self.router.post("/set")(self.set)

    def construct(
        self, data: ChatHistoryItemFactoryRequest.Construct
    ) -> ChatHistoryItemFactoryResponse.Construct:
        chat_history_item = self._chat_history_item_factory.construct(
            **data.model_dump()
        )
        return ChatHistoryItemFactoryResponse.Construct(
            chat_history_item=chat_history_item
        )

    def get_chat_history_item_dict_deep_copy(
        self,
    ) -> ChatHistoryItemFactoryResponse.GetChatHistoryItemDictDeepCopy:
        chat_history_item_dict = (
            self._chat_history_item_factory.get_chat_history_item_dict_deep_copy()
        )
        return ChatHistoryItemFactoryResponse.GetChatHistoryItemDictDeepCopy(
            chat_history_item_dict=chat_history_item_dict
        )

    def set(self, data: ChatHistoryItemFactoryRequest.Set) -> None:
        self._chat_history_item_factory.set(**data.model_dump())
        return

    @staticmethod
    def start_server(
        chat_history_item_factory: ChatHistoryItemFactory, port: int, prefix: str
    ) -> None:
        app = FastAPI()
        router = APIRouter()
        _ = ChatHistoryItemFactoryServer(router, chat_history_item_factory)
        app.include_router(router, prefix=prefix)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_config=None,  # Disable Uvicorn logging
        )
