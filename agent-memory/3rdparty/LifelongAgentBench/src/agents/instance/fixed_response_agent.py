import json

from src.agents.agent import Agent
from src.typings import (
    ChatHistoryItem,
    AgentUnknownException,
    Role,
    ChatHistory,
    Session,
)


class FixedResponseAgent(Agent):
    """
    The agent will read run history and generate fixed response. It is used for reproducing the bug.
    """

    def __init__(self, session_history_file_path: str):
        self.session_list: list[Session] = [
            Session.model_validate(session_dict)
            for session_dict in json.load(open(session_history_file_path))
        ]

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        for session in self.session_list:
            session_chat_history_length = session.chat_history.get_value_length()
            # The first chat history item is the: task_requirement (+ experiences) + instruction
            # If the PreviousSampleUtilization Callback is enabled for the original experiment, The direct comparison
            #   of the first chat history item will not work, due to the fact that the recovered experiment has no
            #   experiences.
            for item_index in range(1, chat_history.get_value_length()):
                if item_index >= session_chat_history_length:
                    break
                session_chat_history_item = session.chat_history.get_item_deep_copy(
                    item_index
                )
                input_chat_history_item = chat_history.get_item_deep_copy(item_index)
                if session_chat_history_item != input_chat_history_item:
                    break
            else:
                item_index = chat_history.get_value_length()
                if item_index >= session_chat_history_length:
                    raise AgentUnknownException(
                        "FixedResponseAgent cannot find response for the given chat history."
                    )
                return ChatHistoryItem(
                    role=Role.AGENT,
                    content=session.chat_history.get_item_deep_copy(item_index).content,
                )
        raise AgentUnknownException(
            "FixedResponseAgent cannot find response for the given chat history."
        )
