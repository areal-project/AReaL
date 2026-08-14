import json
from typing import Optional
import re

from src.agents.agent import Agent
from src.typings import (
    ChatHistoryItem,
    AgentUnknownException,
    Role,
    TaskName,
    ChatHistory,
)


class OracleAgent(Agent):
    """
    The agent will generate oracle response based on the information of dataset.
    There is no inherent relationship between the OracleAgent and Agent that may be called as DBBenchOracleAgent. This
    is just because I want to keep the agent simple and easy to read. It is only used for debugging.
    """

    def __init__(self, task_name: TaskName, data_file_path: str):
        self.task_name = task_name
        self.response_dict: dict[str, list[str]] = {}
        data_dict = json.load(open(data_file_path, "r"))
        for entry in data_dict.values():
            match self.task_name:
                case TaskName.DB_BENCH:
                    action_response = (
                        f"Think: Hello world. "
                        f"Action: Operation\n```sql\n{entry['sql']}\n```"
                    )
                    final_answer: str
                    if entry.get("answer_direct") is not None:
                        processed_answer_direct = [
                            tuple(row) for row in entry["answer_direct"]
                        ]
                        final_answer = str(processed_answer_direct)
                    else:
                        final_answer = entry["answer_md5"]
                    finish_response = f"Think: Hello world. Action: Answer\nFinal Answer: {final_answer}"
                    self.response_dict[entry["instruction"]] = [
                        action_response,
                        finish_response,
                    ]
                case TaskName.OS_INTERACTION:
                    action_response = (
                        f"Think: Hello world. "
                        f"Act: bash \n```bash\n{entry['evaluation_info']['ground_truth_command_item']['script']}\n```"
                    )
                    finish_response = f"Think: Hello world. Act: finish."
                    self.response_dict[entry["instruction"]] = [
                        action_response,
                        finish_response,
                    ]
                case TaskName.KNOWLEDGE_GRAPH:
                    entity_dict: dict[str, str] = entry["entity_dict"]
                    response_list: list[str] = []
                    for action in entry["action_list"]:
                        for key, value in entity_dict.items():
                            action = action.replace(value, key)
                        response_list.append(f"Thought: Hello world. Action: {action}")
                    self.response_dict[entry["question"]] = response_list
                    response_list.append(
                        "Thought: Hello world. Final Answer: #<<VARIABLE_INDEX>>"
                    )
                case _:
                    raise NotImplementedError()

    def _inference(self, chat_history: ChatHistory) -> ChatHistoryItem:
        # The function involves lots of nesting for loops, which is not recommended.
        # But it is only used for debugging, so it is acceptable.
        # region Get chat_corresponding_instruction
        chat_corresponding_instruction: Optional[str] = None
        for item_index in range(chat_history.get_value_length()):
            content = chat_history.get_item_deep_copy(item_index).content
            for instruction in self.response_dict.keys():
                if instruction in content:
                    if chat_corresponding_instruction is None:
                        chat_corresponding_instruction = instruction
                    elif len(instruction) > len(chat_corresponding_instruction):
                        # Match longer instruction.
                        chat_corresponding_instruction = instruction
                    elif len(instruction) == len(chat_corresponding_instruction):
                        raise AgentUnknownException(
                            "Duplicate instruction. Check the data."
                        )
            if chat_corresponding_instruction is not None:
                break
        if chat_corresponding_instruction is None:
            raise AgentUnknownException(
                "OracleAgent cannot find response for the given chat history."
            )
        response_list = self.response_dict[chat_corresponding_instruction]
        # endregion
        # region Get current_response_index and current_response
        current_response_index: Optional[int] = None
        for response_index, response in enumerate(response_list):
            for item_index in range(chat_history.get_value_length()):
                chat_history_item = chat_history.get_item_deep_copy(item_index)
                if chat_history_item.role == Role.USER:
                    continue
                if response == chat_history_item.content:
                    current_response_index = response_index + 1
        if current_response_index is None:
            current_response_index = 0
        current_response = response_list[current_response_index]
        # endregion
        match self.task_name:
            case TaskName.DB_BENCH:
                return ChatHistoryItem(role=Role.AGENT, content=current_response)
            case TaskName.OS_INTERACTION:
                return ChatHistoryItem(role=Role.AGENT, content=current_response)
            case TaskName.KNOWLEDGE_GRAPH:
                if current_response_index == len(response_list) - 1:
                    last_content = chat_history.get_item_deep_copy(-1).content
                    match = re.search(r"Variable #(\d+)", last_content)
                    if match:
                        variable_index = int(match.group(1))
                    else:
                        variable_index = 0  # Default value
                    return ChatHistoryItem(
                        role=Role.AGENT,
                        content=response_list[current_response_index].replace(
                            "<<VARIABLE_INDEX>>", str(variable_index)
                        ),
                    )
                else:
                    return ChatHistoryItem(role=Role.AGENT, content=current_response)
            case _:
                raise NotImplementedError()
