import os
import yaml
import re

from src.typings import TaskName
from src.utils import ConfigLoader

HUGGINGFACE_GREEDY_DECODING_CONFIG_DICT = {
    # https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/text_generation#transformers.GenerationConfig.generation_kwargs
    # Greedy decoding
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 512,  # The same as AgentBench
    "temperature": None,
    "top_k": None,
    "top_p": None,
}


class ConfigFactory:
    def __init__(self, definition_path: str, output_root_dir: str) -> None:
        self.definition_path = definition_path
        self.output_root_dir = output_root_dir
        assert os.path.exists(self.output_root_dir)

    @staticmethod
    def extract_number_from_str(pattern: str, _str: str) -> int:
        number_match = re.search(pattern, _str)
        assert number_match is not None
        number = int(number_match.group(1))
        return number

    def create_config(
        self, agent_language_model_name: str, task_name: TaskName, config_file_name: str
    ) -> None:
        definition_dict = ConfigLoader().load_from(self.definition_path)
        assert agent_language_model_name in definition_dict["language_model_dict"]
        task_name_str: str = task_name.value  # noqa
        # region Write agent_config
        agent_config_dir = (
            agent_language_model_name.replace("-", "_").replace(".", "").lower()
        )
        agent_config_output_dir = os.path.join(self.output_root_dir, agent_config_dir)
        agent_config_output_path = os.path.join(agent_config_output_dir, "agent.yaml")
        if not os.path.exists(agent_config_output_dir):
            os.makedirs(agent_config_output_dir)
        agent_config_dict = {
            "assignment_config": {
                "language_model_list": [{"name": agent_language_model_name}],
                "agent": {
                    "name": "language_model_agent",
                    "custom_parameters": {
                        "language_model": agent_language_model_name,
                        "inference_config_dict": HUGGINGFACE_GREEDY_DECODING_CONFIG_DICT,
                    },
                },
            }
        }
        if not os.path.exists(agent_config_output_path):
            yaml.dump(
                agent_config_dict, open(agent_config_output_path, "w"), sort_keys=False
            )
        else:
            original_agent_config_dict = yaml.safe_load(
                open(agent_config_output_path, "r")
            )
            assert original_agent_config_dict == agent_config_dict
        # endregion
        # region Write task_config
        task_config_output_dir = os.path.join(
            agent_config_output_dir, "instance", task_name_str
        )
        task_config_output_path = os.path.join(task_config_output_dir, "task.yaml")
        task_config_dict = {"assignment_config": {"task": task_name_str}}
        if not os.path.exists(task_config_output_dir):
            os.makedirs(task_config_output_dir)
        if not os.path.exists(task_config_output_path):
            yaml.dump(
                task_config_dict, open(task_config_output_path, "w"), sort_keys=False
            )
        else:
            original_task_config_dict = yaml.safe_load(
                open(task_config_output_path, "r")
            )
            assert original_task_config_dict == task_config_dict
        # endregion
        # region Write experiment_config
        if config_file_name.startswith("standard"):
            experiment_config = {
                "import": [
                    "../task.yaml",
                    "../../../agent.yaml",
                    "../../../../../../definition.yaml",
                ],
                "assignment_config": {
                    "callback_dict": {
                        "callback_0": {"name": "current_session_saving_callback"},
                        "callback_1": {
                            "name": "consecutive_abnormal_agent_inference_process_handling_callback"
                        },
                    },
                    "output_dir": "outputs/{TIMESTAMP}",
                    "sample_order": "default",
                },
                "environment_config": {"use_task_client_flag": True},
            }
        elif config_file_name.startswith("previous_sample_utilization"):
            chat_history_item_dict_path = os.path.join(
                "./chat_history_items/previous_sample_utilization",
                f"{task_name_str}.json",
            )
            utilized_sample_count = ConfigFactory.extract_number_from_str(
                r"previous_sample_utilization_usc(\d+)\.yaml", config_file_name
            )
            experiment_config = {
                "import": [
                    "../task.yaml",
                    "../../../agent.yaml",
                    "../../../../../../definition.yaml",
                ],
                "assignment_config": {
                    "callback_dict": {
                        "callback_0": {"name": "current_session_saving_callback"},
                        "callback_1": {
                            "name": "consecutive_abnormal_agent_inference_process_handling_callback"
                        },
                        "callback_2": {
                            "name": "previous_sample_utilization_callback",
                            "custom_parameters": {
                                "utilized_sample_count": utilized_sample_count
                            },
                        },
                    },
                    "output_dir": "outputs/{TIMESTAMP}",
                    "sample_order": "default",
                },
                "environment_config": {"use_task_client_flag": True},
                "task_dict": {
                    task_name_str: {
                        "parameters": {
                            "chat_history_item_factory": {
                                "parameters": {
                                    "chat_history_item_dict_path": chat_history_item_dict_path
                                }
                            }
                        }
                    }
                },
            }
        elif config_file_name.startswith("group_self_consistency"):
            group_count = ConfigFactory.extract_number_from_str(
                r"group_self_consistency_gc(\d+)_scpg\d+\.yaml", config_file_name
            )
            sample_count_per_group = ConfigFactory.extract_number_from_str(
                r"group_self_consistency_gc\d+_scpg(\d+)\.yaml", config_file_name
            )
            relevance_judgement_batch_size: int
            experience_utilization_batch_size: int
            model_size = ConfigFactory.extract_number_from_str(
                r"-(\d+)B-Instruct", agent_language_model_name
            )
            if model_size in [7, 8]:
                relevance_judgement_batch_size = 64
                experience_utilization_batch_size = 32
            elif model_size in [70, 72]:
                relevance_judgement_batch_size = 16
                experience_utilization_batch_size = 8
            else:
                raise NotImplementedError()
            experiment_config = {
                "import": [
                    "../task.yaml",
                    "../../../agent.yaml",
                    "../../../../../../definition.yaml",
                ],
                "assignment_config": {
                    "callback_dict": {
                        "callback_0": {"name": "current_session_saving_callback"},
                        "callback_1": {
                            "name": "consecutive_abnormal_agent_inference_process_handling_callback"
                        },
                        "callback_2": {
                            "name": "group_self_consistency_callback",
                            "custom_parameters": {
                                "group_count": group_count,
                                "sample_count_per_group": sample_count_per_group,
                                "batch_size_dict": {
                                    "relevance_judgement": relevance_judgement_batch_size,
                                    "experience_utilization": experience_utilization_batch_size,
                                },
                                "language_model": agent_language_model_name,
                                "task_name": {
                                    "module": "src.typings.TaskName",
                                    "parameters": {"value": "db_bench"},
                                },
                                "inference_config_dict": HUGGINGFACE_GREEDY_DECODING_CONFIG_DICT,
                            },
                        },
                    },
                    "output_dir": "outputs/{TIMESTAMP}",
                    "sample_order": "default",
                },
                "environment_config": {"use_task_client_flag": True},
            }
        else:
            raise NotImplementedError()
        experiment_config_output_dir = os.path.join(task_config_output_dir, "instance")
        experiment_config_output_path = os.path.join(
            experiment_config_output_dir, config_file_name
        )
        if not os.path.exists(experiment_config_output_dir):
            os.makedirs(experiment_config_output_dir)
        if not os.path.exists(experiment_config_output_path):
            yaml.dump(
                experiment_config,
                open(experiment_config_output_path, "w"),
                sort_keys=False,
            )
        else:
            original_experiment_config = yaml.safe_load(
                open(experiment_config_output_path, "r")
            )
            assert original_experiment_config == experiment_config
        # endregion


def main() -> None:
    config_factory = ConfigFactory(
        "configs/definition.yaml", "configs/assignments/experiments"
    )
    for agent_language_model_name in [
        "Llama-3.1-8B-Instruct",
        "DeepSeek-R1-Distill-Llama-8B",
        "Qwen2.5-7B-Instruct",
        "DeepSeek-R1-Distill-Qwen-7B",
        "Qwen2.5-32B-Instruct",
        "QwQ-32B",
        "Llama-3.1-70B-Instruct",
        "DeepSeek-R1-Distill-Llama-70B",
        "Qwen2.5-72B-Instruct",
        "DeepSeek-R1-Distill-Qwen-32B",
    ]:
        for task_name in [
            TaskName.DB_BENCH,
            TaskName.OS_INTERACTION,
            TaskName.KNOWLEDGE_GRAPH,
        ]:
            for config_file_name in [
                "standard.yaml",
                "previous_sample_utilization_usc1.yaml",
                "previous_sample_utilization_usc4.yaml",
                "previous_sample_utilization_usc16.yaml",
                "previous_sample_utilization_usc32.yaml",
                "previous_sample_utilization_usc64.yaml",
                # "group_self_consistency_gc1_scpg1.yaml",
                # "group_self_consistency_gc2_scpg2.yaml",
                # "group_self_consistency_gc4_scpg1.yaml",
                # "group_self_consistency_gc2_scpg4.yaml",
                # "group_self_consistency_gc4_scpg2.yaml",
                # "group_self_consistency_gc8_scpg1.yaml",
                # "group_self_consistency_gc4_scpg16.yaml",
                # "group_self_consistency_gc16_scpg4.yaml",
                # "group_self_consistency_gc64_scpg1.yaml",
            ]:
                config_factory.create_config(
                    agent_language_model_name, task_name, config_file_name
                )


if __name__ == "__main__":
    main()
