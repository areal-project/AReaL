from pydantic import BaseModel, field_validator
from enum import StrEnum, unique
from typing import Mapping

SampleIndex = int | str
MetricDict = dict[str, dict[str, dict[str, float]]]


@unique
class Role(StrEnum):
    USER = "user"
    AGENT = "agent"


class ChatHistoryItem(BaseModel):
    role: Role
    content: str


class ChatHistoryItemDict(BaseModel):
    value: dict[str, ChatHistoryItem]

    @field_validator("value", mode="before")  # noqa
    @classmethod
    def _ensure_str(
        cls, before: Mapping[str | int, ChatHistoryItem]
    ) -> dict[str, ChatHistoryItem]:
        after = {}
        for k, v in before.items():
            after[str(k)] = v
        return after

    def set_chat_history_item(self, index: int | str, role: Role, content: str) -> None:
        if isinstance(index, int):
            index = str(index)
        if index in self.value.keys():
            assert role == self.value[index].role
        self.value[index] = ChatHistoryItem(role=role, content=content)


class TaskName(StrEnum):
    DB_BENCH = "db_bench"
    OS_INTERACTION = "os_interaction"
    KNOWLEDGE_GRAPH = "knowledge_graph"
