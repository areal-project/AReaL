from typing import Optional, Any, Mapping, Generator
from pydantic import BaseModel
from enum import StrEnum

from .general import SampleIndex, ChatHistoryItem, TaskName, Role
from .status import SampleStatus


class ChatHistory(BaseModel):
    # Comment out __getattribute__ and __setattr__ if bugs in accessing the value appear.
    # But REMEMBER, There are NO situations in which you should access the value directly.
    value: list[ChatHistoryItem] = []  # Should be accessed by other classes

    def inject(self, item: ChatHistoryItem | Mapping[str, Role | str]) -> None:
        value = super().__getattribute__("value")
        if isinstance(item, ChatHistoryItem):
            pass  # Do nothing
        elif isinstance(item, Mapping):
            item = ChatHistoryItem.model_validate(item)
        else:
            raise TypeError(f"Unsupported type {type(item)}")
        if len(value) > 0:
            last_role = value[-1].role
            current_role = item.role
            assert last_role != current_role
        value.append(item)

    def set(
        self, item_index: int, item: ChatHistoryItem | Mapping[str, Role | str]
    ) -> None:
        value = super().__getattribute__("value")
        original_item = value[item_index]
        if isinstance(item, ChatHistoryItem):
            assert original_item.role == item.role
        elif isinstance(item, Mapping):
            assert original_item.role == item["role"]
            item = ChatHistoryItem.model_validate(item)
        else:
            raise TypeError()
        value[item_index] = item

    def pop(self, item_index: int) -> ChatHistoryItem:
        value: list[ChatHistoryItem] = super().__getattribute__("value")
        return value.pop(item_index)

    def get_item_deep_copy(self, item_index: int) -> ChatHistoryItem:
        item = super().__getattribute__("value")[item_index]
        item_copy: ChatHistoryItem = item.model_copy(deep=True)
        return item_copy

    def get_value_length(self) -> int:
        # To better track the usage of this method, we use a method instead of a property.
        return len(super().__getattribute__("value"))

    def get_value_str(
        self,
        role_dict: Mapping[Role, str],
        *,
        start_index: Optional[int],
        end_index: Optional[int],
    ) -> str:
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = self.get_value_length()
        assert start_index < end_index <= self.get_value_length()
        chat_history_item_str_list: list[str] = []
        exist_empty_agent_response_flag = False
        for item_index in range(start_index, end_index):
            chat_history_item = self.get_item_deep_copy(item_index)
            content = chat_history_item.content
            if chat_history_item.role == Role.AGENT and content == "":
                exist_empty_agent_response_flag = True
                break
            chat_history_item_str_list.append(
                f"{role_dict[chat_history_item.role]}: {content}"
            )
        if exist_empty_agent_response_flag:
            # The agent returned an empty response, which indicates that the previous user content (or environment
            # observation) may have been too long, causing the length of the conversation history to exceed the
            # LLM context limit and resulting in an empty response. In this case, even if the user content is added
            # to value_str, the result cannot be utilized by the agent. Therefore, we remove this potentially
            # overlong environment observation here.
            chat_history_item_str_list = chat_history_item_str_list[:-1]
        value_str = "\n".join(chat_history_item_str_list)
        return value_str

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        # https://docs.python.org/3/library/typing.html#annotating-generators-and-coroutines
        raise RuntimeError("The property is disabled.")

    def __getattribute__(self, name: str) -> Any:
        if name == "value":
            raise AttributeError("Cannot get the value directly.")
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "value":
            raise AttributeError("Cannot set the value directly.")
        super().__setattr__(name, value)


class SessionEvaluationOutcome(StrEnum):
    """
    UNKNOWN: Error occurs during evaluation
    IGNORED: The evaluation result cannot be set as a boolean value, the detailed evaluation result is in
      detail_dict. Currently, There is NO situation that the evaluation result is ignored. If the main evaluation
      result is a float value, you should set the status in way like this:
       ```
      status=EvaluationStatus.CORRECT if score == the_upper_bound_of_score else EvaluationStatus.INCORRECT
      ```
      The value is only added for the completeness of the repository.
    """

    UNSET = "unset"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"
    IGNORED = "ignored"

    @staticmethod
    def from_bool(bool_value: bool) -> "SessionEvaluationOutcome":
        if not isinstance(bool_value, bool):
            return SessionEvaluationOutcome.UNKNOWN
        return (
            SessionEvaluationOutcome.CORRECT
            if bool_value
            else SessionEvaluationOutcome.INCORRECT
        )


class SessionEvaluationRecord(BaseModel):
    outcome: SessionEvaluationOutcome = SessionEvaluationOutcome.UNSET
    # Should only be used to record the evaluation result.
    # Should only support following three types
    # Do not initialize it with an empty dict.
    detail_dict: Optional[dict[str, Optional[float | int | bool]]] = None


class Session(BaseModel):
    # self.finish_reason can set for further clarification.
    # self.sample_status can be set to SampleStatus.AGENT_CONTEXT_LIMIT for many reasons,such as network issue,
    # exceed context length, no enough memory.
    # Please, set self.finish_reason when self.sample_status falls into
    #   SampleStatus.TASK_UNKNOWN_ERROR
    #   SampleStatus.AGENT_UNKNOWN_ERROR
    # No matter whether a self.sample_status is SampleStatus.RUNNING or SampleStatus.AGENT_CONTEXT_LIMIT after
    # the agent generates a response, the session will be passed to subclass of src.tasks.Task (Task).
    # If self.sample_status.is_generation_process_abnormal(), Task will be responsible for giving
    # task_output. Then, the code should break the infinite interation loop.
    # If self.sample_status == SampleStatus.COMPLETED, Task will generate user side response. If error happens,
    # such as interaction rounds reach the maximum, Task will set self.SampleStatus to corresponding value (may
    # also set self.finish_reason).
    # The interaction loop will continue if and only if self.sample_status == SampleStatus.RUNNING.
    task_name: TaskName
    sample_index: SampleIndex
    sample_status: SampleStatus = SampleStatus.INITIAL
    chat_history: ChatHistory = ChatHistory()
    finish_reason: Optional[str] = None  # Set for further clarification
    task_output: Optional[dict[str, Optional[str]]] = None  # Answer to user intent
    evaluation_record: SessionEvaluationRecord = SessionEvaluationRecord()


class SessionMetricCalculationPartial(BaseModel):
    # Use list[Session] for Task.calculate_metric() will cause an oversize payload, which is hard to transport through
    # Http. Therefore, we use SessionMetricCalculationPartial to store the key information of the Session.
    # To keep the maintainability of the code, please do not add other fields to this class.
    sample_index: SampleIndex
    sample_status: SampleStatus
    evaluation_record: SessionEvaluationRecord
