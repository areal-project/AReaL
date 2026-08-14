import json
import os

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import SampleStatus, SessionEvaluationOutcome, SampleIndex


class ConsecutiveAbnormalAgentInferenceProcessHandlingCallback(Callback):
    def __init__(self, tolerance_count: int):
        super().__init__()
        assert tolerance_count > 0
        self.tolerance_count = tolerance_count
        self.consecutive_abnormality_count = 0
        self.aborted_sample_index_list: list[SampleIndex] = []

    def _get_consecutive_abnormality_count_state_info(self) -> tuple[str, str]:
        return (
            os.path.join(self.get_state_dir(), "consecutive_abnormality_count.json"),
            "consecutive_abnormality_count",
        )

    def _get_aborted_sample_index_list_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "aborted_sample_index_list.json")

    def restore_state(self) -> None:
        (
            consecutive_abnormality_count_state_path,
            consecutive_abnormality_count_state_key,
        ) = self._get_consecutive_abnormality_count_state_info()
        self.consecutive_abnormality_count = json.load(
            open(consecutive_abnormality_count_state_path, "r")
        )[consecutive_abnormality_count_state_key]
        self.aborted_sample_index_list = json.load(
            open(self._get_aborted_sample_index_list_state_path(), "r")
        )

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        if self.consecutive_abnormality_count != self.tolerance_count:
            return
        current_session = callback_args.current_session
        self.aborted_sample_index_list.append(current_session.sample_index)
        current_session.sample_status = SampleStatus.AGENT_UNKNOWN_ERROR
        current_session.finish_reason = (
            f"AbnormalAgentInferenceProcessHandlingCallback: The session is aborted because the agent inference "
            f"process is abnormal consecutively for {self.tolerance_count} times."
        )
        # Do not need to set should_agent_inference and should_task_interact to False.
        callback_args.session_controller.should_task_reset = False
        callback_args.session_controller.should_task_complete = False
        current_session.evaluation_record.outcome = SessionEvaluationOutcome.INCORRECT

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        if (
            callback_args.current_session.sample_status.is_agent_inference_process_abnormal()
        ):
            self.consecutive_abnormality_count += 1
        else:
            self.consecutive_abnormality_count = 0

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        (
            consecutive_abnormality_count_state_path,
            consecutive_abnormality_count_state_key,
        ) = self._get_consecutive_abnormality_count_state_info()
        json.dump(
            {
                consecutive_abnormality_count_state_key: self.consecutive_abnormality_count
            },
            open(consecutive_abnormality_count_state_path, "w"),  # noqa
            indent=2,
        )
        json.dump(
            self.aborted_sample_index_list,
            open(self._get_aborted_sample_index_list_state_path(), "w"),  # noqa
            indent=2,
        )
