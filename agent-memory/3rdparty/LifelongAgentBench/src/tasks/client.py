from typing import Sequence

from src.typings import (
    TaskResponse,
    TaskRequest,
    Session,
    SampleIndex,
    MetricDict,
    SessionMetricCalculationPartial,
)
from src.utils import Client
from .task import TaskInterface


class TaskClient(Client, TaskInterface):
    def __init__(self, server_address: str, request_timeout: int):
        Client.__init__(
            self, server_address=server_address, request_timeout=request_timeout
        )

    def get_sample_index_list(self) -> list[SampleIndex]:
        response: TaskResponse.GetSampleIndexList = self._call_server(
            "/get_sample_index_list", None, TaskResponse.GetSampleIndexList
        )
        return response.sample_index_list

    def reset(self, session: Session) -> None:
        response: TaskResponse.Reset = self._call_server(
            "/reset", TaskRequest.Reset(session=session), TaskResponse.Reset
        )
        session.__dict__.update(response.session.__dict__)  # In-place update

    def interact(self, session: Session) -> None:
        response: TaskResponse.Interact = self._call_server(
            "/interact",
            TaskRequest.Interact(session=session),
            TaskResponse.Interact,
        )
        session.__dict__.update(response.session.__dict__)

    def complete(self, session: Session) -> None:
        response: TaskResponse.Complete = self._call_server(
            "/complete",
            TaskRequest.Complete(session=session),
            TaskResponse.Complete,
        )
        session.__dict__.update(response.session.__dict__)

    def release(self) -> None:
        _ = self._call_server(
            "/release",
            None,
            None,
        )

    def calculate_metric(
        self, session_partial_list: Sequence[SessionMetricCalculationPartial]
    ) -> MetricDict:
        response: TaskResponse.CalculateMetric = self._call_server(
            "/calculate_metric",
            TaskRequest.CalculateMetric(session_partial_list=session_partial_list),
            TaskResponse.CalculateMetric,
        )
        return response.metric
