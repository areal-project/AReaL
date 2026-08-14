from typing import Sequence, Mapping, final
from typing_extensions import override, Optional
from abc import ABC, abstractmethod
import os

from src.tasks import Task, DatasetItem
from src.agents import Agent
from src.typings import Session


class SessionContext:
    def __init__(
        self, task: Task[DatasetItem], agent: Agent, session_list: Sequence[Session]
    ):
        self.task = task
        self.agent = agent
        # Please make the session_list to be a read-only list.
        self.__session_list = session_list

    def get_session_list_deep_copy(self) -> list[Session]:
        return [session.model_copy(deep=True) for session in self.__session_list]


class SessionController:
    def __init__(self) -> None:
        self.should_task_reset: bool = True
        self.should_agent_inference: bool = True
        self.should_task_interact: bool = True
        self.should_task_complete: bool = True

    def reset(self) -> None:
        self.should_task_reset = True
        self.should_agent_inference = True
        self.should_task_interact = True
        self.should_task_complete = True


class CallbackArguments:
    def __init__(
        self,
        current_session: Session,
        task: Task[DatasetItem],
        agent: Agent,
        session_list: Sequence[Session],
    ):
        self.current_session = current_session
        self.session_context = SessionContext(
            task=task, agent=agent, session_list=session_list
        )
        self.session_controller = SessionController()


class Callback(ABC):
    """
    A clumsy imitation of transformers.TrainerCallback
    """

    def __init__(self) -> None:
        self.__state_dir: Optional[str] = None

    @final
    def set_state_dir(self, state_dir: str) -> None:
        assert self.__state_dir is None
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
        self.__state_dir = state_dir

    @final
    def get_state_dir(self) -> str:
        assert self.__state_dir is not None
        return self.__state_dir

    @classmethod
    @abstractmethod
    def is_unique(cls) -> bool:
        raise NotImplementedError()

    def restore_state(self) -> None:
        pass

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        pass

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        pass

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        pass

    def on_task_interact(self, callback_args: CallbackArguments) -> None:
        pass

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        pass

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        pass


class CallbackHandler(Callback):
    """Internal class that just calls the callbacks in order."""

    def __init__(self, callback_dict: Mapping[str, Callback]):
        super().__init__()
        self.callback_dict = callback_dict

    @classmethod
    @override
    def is_unique(cls) -> bool:
        raise RuntimeError(
            "The singleton_flag of the CallbackHandler should never be checked."
        )

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        self._call_event("on_session_create", callback_args)

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        self._call_event("on_task_reset", callback_args)

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        self._call_event("on_agent_inference", callback_args)

    def on_task_interact(self, callback_args: CallbackArguments) -> None:
        self._call_event("on_task_interact", callback_args)

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        self._call_event("on_task_complete", callback_args)

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        self._call_event("on_state_save", callback_args)

    def _call_event(self, event: str, callback_args: CallbackArguments) -> None:
        for callback_id, callback in self.callback_dict.items():
            # callback_id is not used. But it can be used for debugging.
            getattr(callback, event)(callback_args)
