import os
from typing import Optional
import json

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import (
    Session,
    Role,
    SessionEvaluationOutcome,
    SampleStatus,
    ChatHistoryItem,
)


class PreviousSampleUtilizationCallback(Callback):
    def __init__(
        self,
        original_first_user_prompt: str,
        utilized_sample_count: int,
    ):
        super().__init__()
        self.original_first_user_prompt = original_first_user_prompt
        self.pattern = "{previous_sample_utilization_target_position}"
        assert self.original_first_user_prompt.count(self.pattern) == 1
        assert utilized_sample_count > 0
        self.utilized_sample_count = utilized_sample_count
        self.utilized_session_list: list[Session] = []

    def _get_utilized_session_list_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "utilized_session_list.json")

    def restore_state(self) -> None:
        self.utilized_session_list = [
            Session.model_validate(session_info_dict)
            for session_info_dict in json.load(
                open(self._get_utilized_session_list_state_path(), "r")
            )
        ]

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        # Get the session that just completed.
        current_session = callback_args.current_session
        if (
            current_session.evaluation_record.outcome
            == SessionEvaluationOutcome.CORRECT
            and current_session.sample_status == SampleStatus.COMPLETED
        ):
            self.utilized_session_list.append(current_session)
        if len(self.utilized_session_list) > self.utilized_sample_count:
            self.utilized_session_list.pop(0)

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        # The session is just created, so its chat_history should be empty.
        assert callback_args.current_session.chat_history.get_value_length() == 0
        # Step1. Construct example_text.
        agent_role_dict = callback_args.session_context.agent.get_role_dict()
        example_text = "\n"
        for i, session in enumerate(self.utilized_session_list):
            try:
                question = session.chat_history.get_item_deep_copy(2).content
            except:  # noqa
                question = ""
            session_str = f"Question {question}:\n"
            session_str += session.chat_history.get_value_str(
                agent_role_dict, start_index=3, end_index=None
            )
            example_text += session_str
        # Step2. Replace the pattern with the example_text.
        first_user_prompt = self.original_first_user_prompt.replace(
            self.pattern, example_text
        )
        callback_args.session_context.task.chat_history_item_factory.set(
            0, Role.USER, first_user_prompt
        )

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        last_chat_history_item = (
            callback_args.current_session.chat_history.get_item_deep_copy(-1)
        )
        assert last_chat_history_item.role == Role.AGENT
        last_agent_response = last_chat_history_item.content
        counterfeit_user_response_location = last_agent_response.find("\nuser: ")
        if counterfeit_user_response_location != -1:
            last_agent_response = last_agent_response[
                :counterfeit_user_response_location
            ]
        callback_args.current_session.chat_history.set(
            -1,
            ChatHistoryItem(
                role=Role.AGENT,
                content=last_agent_response,
            ),
        )

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        json.dump(
            [s.model_dump() for s in self.utilized_session_list],
            open(self._get_utilized_session_list_state_path(), "w"),  # noqa
            indent=2,
        )
