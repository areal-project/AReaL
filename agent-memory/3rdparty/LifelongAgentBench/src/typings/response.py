from pydantic import BaseModel
from typing import Optional, Any

from .session import Session
from .general import SampleIndex, ChatHistoryItem, ChatHistoryItemDict, MetricDict
from .instance_factory import InstanceFactoryType


class GeneralResponse:
    class GetAttribute(BaseModel):
        instance_factory_type: Optional[InstanceFactoryType]
        instance_factory_parameter_dict: Optional[dict[str, Any]]

    class Ping(BaseModel):
        response: str


class TaskResponse:
    class GetSampleIndexList(BaseModel):
        sample_index_list: list[SampleIndex]

    class Reset(BaseModel):
        session: Session

    class Interact(BaseModel):
        session: Session

    class Complete(BaseModel):
        session: Session

    class CalculateMetric(BaseModel):
        metric: MetricDict


class ChatHistoryItemFactoryResponse:
    class Construct(BaseModel):
        chat_history_item: ChatHistoryItem

    class GetChatHistoryItemDictDeepCopy(BaseModel):
        chat_history_item_dict: ChatHistoryItemDict
