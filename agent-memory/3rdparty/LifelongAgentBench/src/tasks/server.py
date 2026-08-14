from fastapi import FastAPI, APIRouter
import uvicorn

from .task import Task, DatasetItem
from src.typings import TaskRequest, TaskResponse
from src.utils import Server


class TaskServer(Server):
    def __init__(self, router: APIRouter, task: Task[DatasetItem]) -> None:
        Server.__init__(self, router, task)
        self.task = task
        self.router.post("/get_sample_index_list")(self.get_sample_index_list)
        self.router.post("/reset")(self.reset)
        self.router.post("/interact")(self.interact)
        self.router.post("/complete")(self.complete)
        self.router.post("/release")(self.release)
        self.router.post("/calculate_metric")(self.calculate_metric)

    def get_sample_index_list(self) -> TaskResponse.GetSampleIndexList:
        sample_index_list = self.task.get_sample_index_list()
        return TaskResponse.GetSampleIndexList(sample_index_list=sample_index_list)

    def reset(self, data: TaskRequest.Reset) -> TaskResponse.Reset:
        self.task.reset(data.session)
        return TaskResponse.Reset(session=data.session)

    def interact(self, data: TaskRequest.Interact) -> TaskResponse.Interact:
        self.task.interact(data.session)
        return TaskResponse.Interact(session=data.session)

    def complete(self, data: TaskRequest.Complete) -> TaskResponse.Complete:
        self.task.complete(data.session)
        return TaskResponse.Complete(session=data.session)

    def release(self) -> None:
        self.task.release()
        return

    def calculate_metric(
        self, data: TaskRequest.CalculateMetric
    ) -> TaskResponse.CalculateMetric:
        metric = self.task.calculate_metric(data.session_partial_list)
        return TaskResponse.CalculateMetric(metric=metric)

    def shutdown(self) -> None:
        self.release()

    @staticmethod
    def start_server(task: Task[DatasetItem], port: int, prefix: str) -> None:
        app = FastAPI()
        router = APIRouter()
        # Create an instance to access the shutdown method
        server_instance = TaskServer(router, task)
        app.include_router(router, prefix=prefix)
        # Add the shutdown event handler using lifespan events
        # https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated
        app.add_event_handler("shutdown", server_instance.shutdown)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_config=None,
        )
