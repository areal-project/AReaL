import json
import os

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import Session, Role


class CurrentSessionSavingCallback(Callback):
    def __init__(self, saving_path: str):
        super().__init__()
        self.saving_path = saving_path
        assert self.saving_path.endswith(".json")
        parent_dir_name = os.path.dirname(self.saving_path)
        if not os.path.exists(parent_dir_name):
            os.makedirs(parent_dir_name)

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        self._save_session(callback_args.current_session)

    def on_task_reset(self, callback_args: CallbackArguments) -> None:
        self._save_session(callback_args.current_session)

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        self._save_session(callback_args.current_session)

    def on_task_interact(self, callback_args: CallbackArguments) -> None:
        self._save_session(callback_args.current_session)

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        self._save_session(callback_args.current_session)

    def _save_session(self, session: Session) -> None:
        # current_session does not need to be reloaded when resuming the experiment, so it is not deemed as a state.
        #   This is the reason why the saving_path is passed as a parameter to the constructor, instead of using a path
        #   in the state directory.
        json.dump(session.model_dump(), open(self.saving_path, "w"), indent=2)  # noqa
