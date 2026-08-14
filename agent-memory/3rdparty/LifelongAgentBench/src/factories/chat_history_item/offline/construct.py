import os

from src.typings import ChatHistoryItemDict, TaskName
from src.factories.chat_history_item.offline.task_requirement import (
    TASK_REQUIREMENT_DICT,
)


def construct_offline(root_path: str, requirement_suffix: str) -> None:
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    for task_name in TaskName:
        chat_history_item_dict = ChatHistoryItemDict(
            value={
                0: {  # type: ignore[dict-item]
                    "role": "user",
                    "content": f"""{TASK_REQUIREMENT_DICT[task_name]}{requirement_suffix}""",
                },
                1: {"role": "agent", "content": "OK."},  # type: ignore[dict-item]
            }
        )
        with open(os.path.join(root_path, f"{task_name}.json"), "w") as f:
            f.write(chat_history_item_dict.model_dump_json(indent=2))


def main() -> None:
    standard_requirement_suffix = """

Now, I will give you the question that you need to solve."""
    previous_sample_utilization_requirement_suffix = """

{previous_sample_utilization_target_position}

Now, I will give you the question that you need to solve."""
    for root_path, requirement_suffix in [
        ("chat_history_items/standard", standard_requirement_suffix),
        (
            "chat_history_items/previous_sample_utilization",
            previous_sample_utilization_requirement_suffix,
        ),
    ]:
        construct_offline(root_path, requirement_suffix)


if __name__ == "__main__":
    main()
