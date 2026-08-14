from typing import final
from abc import ABC, abstractmethod
from typing import Mapping

from src.typings import (
    ChatHistoryItem,
    AgentException,
    AgentUnknownException,
    SampleStatus,
    AgentContextLimitException,
    ChatHistory,
    Role,
    Session,
    AgentOutOfMemoryException,
)


class Agent(ABC):
    @final
    def inference(self, session: Session) -> None:
        # The function takes Session as input for better exception handling
        chat_history = session.chat_history
        assert chat_history.get_item_deep_copy(-1).role == Role.USER
        try:
            chat_history_item = self._inference(chat_history)
        except AgentException as e:
            session.finish_reason = str(e)
            if isinstance(e, AgentContextLimitException):
                session.sample_status = SampleStatus.AGENT_CONTEXT_LIMIT
            elif isinstance(e, AgentOutOfMemoryException):
                session.sample_status = SampleStatus.AGENT_OUT_OF_MEMORY
            elif isinstance(e, AgentUnknownException):
                session.sample_status = SampleStatus.AGENT_UNKNOWN_ERROR
            else:
                raise TypeError(
                    f"Please handle {e.__class__.__name__} in Agent.inference()."
                )
            chat_history_item = ChatHistoryItem(role=Role.AGENT, content="")
        except Exception as e:
            session.finish_reason = str(AgentUnknownException.from_exception(e))
            session.sample_status = SampleStatus.AGENT_UNKNOWN_ERROR
            chat_history_item = ChatHistoryItem(role=Role.AGENT, content="")
        session.chat_history.inject(chat_history_item)

    @abstractmethod
    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        # The function takes list[ChatHistoryItem] instead of Session as input to keep the modularity of the code
        raise NotImplementedError()

    def get_role_dict(self) -> Mapping[Role, str]:
        return {role: "dummy" for role in Role}
