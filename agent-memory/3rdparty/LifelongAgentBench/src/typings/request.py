from pydantic import BaseModel
from typing import Optional, Any, Sequence

from .session import Session, SessionMetricCalculationPartial
from .general import Role
from .instance_factory import InstanceFactoryType


class GeneralRequest:
    class GetAttribute(BaseModel):
        name: str

    class SetAttribute(BaseModel):
        name: str
        instance_factory_type: InstanceFactoryType
        instance_factory_parameter_dict: dict[str, Any]


class TaskRequest:
    class Reset(BaseModel):
        session: Session

    class Interact(BaseModel):
        session: Session

    class Complete(BaseModel):
        session: Session

    class CalculateMetric(BaseModel):
        session_partial_list: Sequence[SessionMetricCalculationPartial]


class ChatHistoryItemFactoryRequest:
    class Construct(BaseModel):
        chat_history_item_index: int
        expected_role: Optional[Role]

    class Set(BaseModel):
        prompt_index: int
        role: Role
        content: str
