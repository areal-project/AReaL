import os
from typing import Mapping

from src.tasks import Task, DatasetItem
from src.language_models import LanguageModel
from src.agents import Agent
from src.typings import AssignmentConfig, Role
from .instance import *
from .callback import Callback


class CallbackConstructor:
    @staticmethod
    def construct(
        assignment_config: AssignmentConfig,
        task: Task[DatasetItem],
        agent: Agent,
        language_model_dict: Mapping[str, LanguageModel],
    ) -> dict[str, Callback]:
        instanced_unique_callback_name_set: set[str] = set()
        for (
            callback_id,
            general_instance_factory,
        ) in assignment_config.callback_dict.items():
            target_class_str = general_instance_factory.module.split(".")[-1]
            unique_flag: bool
            match target_class_str:
                case PreviousSampleUtilizationCallback.__name__:
                    unique_flag = PreviousSampleUtilizationCallback.is_unique()
                    first_chat_history_item = task.chat_history_item_factory.construct(
                        0, expected_role=Role.USER
                    )
                    general_instance_factory.parameters[
                        "original_first_user_prompt"
                    ] = first_chat_history_item.content
                case CurrentSessionSavingCallback.__name__:
                    unique_flag = CurrentSessionSavingCallback.is_unique()
                    saving_path = os.path.join(
                        assignment_config.output_dir, "current_session.json"
                    )
                    general_instance_factory.parameters["saving_path"] = saving_path
                case GroupSelfConsistencyCallback.__name__:
                    unique_flag = GroupSelfConsistencyCallback.is_unique()
                    general_instance_factory.parameters["language_model"] = (
                        language_model_dict[
                            general_instance_factory.parameters["language_model"]
                        ]
                    )
                case ConsecutiveAbnormalAgentInferenceProcessHandlingCallback.__name__:
                    unique_flag = (
                        ConsecutiveAbnormalAgentInferenceProcessHandlingCallback.is_unique()
                    )
                case _:
                    raise NotImplementedError(
                        f"Callback {target_class_str} is not implemented or not handled in CallbackConstructor."
                    )
            if unique_flag:
                assert target_class_str not in instanced_unique_callback_name_set
                instanced_unique_callback_name_set.add(target_class_str)
        callback_dict: dict[str, Callback] = {
            callback_id: general_instance_factory.create()
            for callback_id, general_instance_factory in assignment_config.callback_dict.items()
        }
        for callback_id, callback in callback_dict.items():
            callback.set_state_dir(
                os.path.join(
                    assignment_config.output_dir, "callback_state", callback_id
                )
            )
        return callback_dict
