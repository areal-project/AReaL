import json
from typing import Optional, Callable, Any, Sequence
import re
from pydantic import field_validator
import inspect

from src.tasks.task import (
    Task,
    DatasetItem,
    SkillUtility,
    AgentResponseParserResult,
    AgentAction,
)
from src.typings import (
    SampleIndex,
    SampleStatus,
    TaskEnvironmentException,
    Session,
    TaskName,
    Role,
    SessionEvaluationOutcome,
    MetricDict,
    SessionMetricCalculationPartial,
)
from src.factories.chat_history_item import ChatHistoryItemFactory
from .api import KnowledgeGraphAPI, Variable, KnowledgeGraphAPIException
from .utils.sparql_executor import SparqlExecutor


class KnowledgeGraphSkillUtility(SkillUtility):
    _SKILL_TO_LEVEL_DICT = {}


class KnowledgeGraphDatasetItem(DatasetItem):
    question: str
    entity_dict: dict[str, str]
    answer_set: set[str]

    @field_validator("entity_dict", mode="before")  # noqa
    @classmethod
    def _validate_parentheses_pair(cls, entity_dict: dict[str, str]) -> dict[str, str]:
        for entity in entity_dict.keys():
            error_message = f"Invalid parentheses pair in entity: {entity}"
            left_parentheses_count = 0
            for char in entity:
                if char == "(":
                    left_parentheses_count += 1
                elif char == ")":
                    left_parentheses_count -= 1
                    if left_parentheses_count < 0:
                        raise ValueError(error_message)
            if left_parentheses_count != 0:
                raise ValueError(error_message)
        return entity_dict

    @field_validator("entity_dict", mode="before")  # noqa
    @classmethod
    def _validate_entity_name(cls, entity_dict: dict[str, str]) -> dict[str, str]:
        for entity in entity_dict.keys():
            if "#" in entity:
                raise ValueError(f"Invalid entity name: {entity}")
        return entity_dict

    def get_skill_list(self) -> list[str]:
        return []

    def get_difficulty_level(self) -> int:
        return 0


class KnowledgeGraph(Task[KnowledgeGraphDatasetItem]):
    def __init__(
        self,
        task_name: TaskName,
        chat_history_item_factory: ChatHistoryItemFactory,
        sparql_url: str,
        ontology_dir_path: str,
        data_file_path: str,
        max_round: int,
    ):
        super().__init__(task_name, chat_history_item_factory, max_round)
        sparql_executor = SparqlExecutor(sparql_url)
        self.knowledge_graph_api = KnowledgeGraphAPI(ontology_dir_path, sparql_executor)
        raw_dataset: dict[str, dict[str, Any]] = json.load(open(data_file_path, "r"))
        dataset: dict[SampleIndex, KnowledgeGraphDatasetItem] = {}
        for key, item in raw_dataset.items():
            question = item["question"]
            entity_dict = item["entity_dict"]
            answer_set = set(item["answer_list"])
            dataset[key] = KnowledgeGraphDatasetItem(
                question=question,
                entity_dict=entity_dict,
                answer_set=answer_set,
            )
        self._set_dataset(dataset)
        self.variable_list: Optional[list[Variable]] = None

    def _get_default_task_output(self) -> dict[str, Optional[str]]:
        return {"answer": None}

    @staticmethod
    def _get_action_pattern() -> str:
        return r"Action: (\w+)\((.+?)\)"

    @staticmethod
    def _extract_argument_str_from_agent_response(agent_response: str) -> Optional[str]:
        # AgentBench returns
        # re.findall(rf"{api_name}\((.+?)\)", agent_response)[0]
        # directly.
        action_pattern = KnowledgeGraph._get_action_pattern()
        argument_list_match = re.search(action_pattern, agent_response)
        if argument_list_match is None:
            return None
        start_index = argument_list_match.start(2) - 1
        left_parentheses_count = 0
        for index in range(start_index, len(agent_response)):
            if agent_response[index] == "(":
                left_parentheses_count += 1
            elif agent_response[index] == ")":
                left_parentheses_count -= 1
                if left_parentheses_count == 0:
                    return agent_response[start_index + 1 : index]
            else:
                continue
        # Incomplete parentheses
        return None

    @staticmethod
    def _extract_argument_list_from_argument_str(
        argument_str: str, entity_list: list[str]
    ) -> list[str]:
        # AgentBench returns re.split(r "\s*,\s*", argument_str) directly.
        if len(argument_str) == 0:
            return []
        for entity in entity_list:
            if "," not in entity:
                continue
            entity_index = argument_str.find(entity)
            if entity_index == -1:
                # entity not found
                continue
            left_str = argument_str[:entity_index]
            left_argument_list = (
                KnowledgeGraph._extract_argument_list_from_argument_str(
                    left_str, entity_list
                )
            )
            right_str = argument_str[entity_index + len(entity) :]
            right_argument_list = (
                KnowledgeGraph._extract_argument_list_from_argument_str(
                    right_str, entity_list
                )
            )
            return left_argument_list + [entity] + right_argument_list
        # no entity found, split in the same way as AgentBench
        argument_list = re.split(r"\s*,\s*", argument_str)
        # ", hello_world" will be split into ["", "hello_world"], remove ""
        argument_list = [argument for argument in argument_list if argument != ""]
        return argument_list

    @staticmethod
    def _extract_variable_index_from_argument(raw_argument: str) -> Optional[int]:
        # The original implementation of AgentBench contains bugs.
        # The method will not check whether the variable index exists.
        possible_lower_prefix_list = ["#", "variable#", "variable #", "var#", "var #"]
        for prefix in possible_lower_prefix_list:
            if raw_argument.lower().startswith(prefix):
                variable_index_str = raw_argument[len(prefix) :]
                try:
                    variable_index = int(variable_index_str)
                    return variable_index
                except Exception as e:
                    raise e
        return None

    @staticmethod
    def _parse_agent_response(agent_response: str) -> AgentResponseParserResult:
        # AgentBench final_answer_pattern: r"Final Answer: #(\d+)"'
        final_answer_pattern = r"Final [Aa]nswer:\s*(?:[Vv]ar(?:iable)?\s*)?#(\d+)"
        action_pattern = KnowledgeGraph._get_action_pattern()
        if (
            final_answer_match := re.search(final_answer_pattern, agent_response)
        ) is not None:
            final_answer = final_answer_match.group(1)
            return AgentResponseParserResult(
                action=AgentAction.FINISH,
                content=final_answer,
                finish_reason=None,
            )
        if (action_match := re.search(action_pattern, agent_response)) is None:
            return AgentResponseParserResult(
                action=AgentAction.INVALID,
                content=None,
                finish_reason=(
                    f"Cannot find the pattern of action in agent response. "
                    f"final_answer_pattern: {final_answer_pattern} "
                    f"action_pattern: {action_pattern}"
                ),
            )
        api_name = action_match.group(1)
        argument_str = KnowledgeGraph._extract_argument_str_from_agent_response(
            agent_response
        )
        if argument_str is None:
            # Analogous to the DBBench environment, it can be regarded as the agent outputting a syntactically
            # incorrect SQL statement.
            return AgentResponseParserResult(
                action=AgentAction.EXECUTE,
                content=f"{api_name}()",  # Cannot find argument list
                finish_reason=None,
            )
        else:
            return AgentResponseParserResult(
                action=AgentAction.EXECUTE,
                content=f"{api_name}({argument_str})",
                finish_reason=None,
            )

    def _get_nonexistent_variable_error_message(self, variable_index: int) -> str:
        error_message = f"Variable #{variable_index} is not found in variable list. "
        assert self.variable_list is not None
        if len(self.variable_list) > 0:
            variable_list_str = "["
            for i in range(len(self.variable_list)):
                variable_list_str += f"#{i}, "
            variable_list_str = variable_list_str[:-2] + "]"
            error_message += f"Current variable list: {variable_list_str}."
        else:
            create_variable_prompt = (
                "The variable list is empty. You can use the following process to create the first variable: "
                "Use get_relations(var: Variable | str) to retrieve the relations connected to the input str or Variable, "
                "then use get_neighbors(var: Variable | str, relation: str) to retrieve the entities connected via the specified relation. "
            )
            error_message += create_variable_prompt
        return error_message

    def _reset(self, session: Session) -> None:
        current_dataset_item: KnowledgeGraphDatasetItem = (
            self._get_current_dataset_item()
        )
        session.chat_history.inject(
            self.chat_history_item_factory.construct(0, expected_role=Role.USER)
        )
        session.chat_history.inject(
            self.chat_history_item_factory.construct(1, expected_role=Role.AGENT)
        )
        question = current_dataset_item.question
        entity_list = list(current_dataset_item.entity_dict.keys())
        session.chat_history.inject(
            {
                "role": Role.USER,
                "content": f"Question: {question}, Entities: {entity_list}",
            }
        )
        self.variable_list = []
        self.knowledge_graph_api.reset_cache()

    def _interact(self, session: Session) -> None:
        # region Parse agent response, ensure the code pass the type check
        parser_result = KnowledgeGraph._parse_agent_response(
            session.chat_history.get_item_deep_copy(-1).content
        )
        assert self.variable_list is not None
        # endregion
        # region Execute action
        match parser_result.action:
            case AgentAction.EXECUTE:
                api_str = parser_result.content
                assert api_str is not None  # Type narrowing
                current_dataset_item = self._get_current_dataset_item()
                # region Get API name
                api_name = api_str.split("(")[0]
                if api_name not in KnowledgeGraphAPI.get_valid_api_name_list():
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": (
                                f"Unknown API name: {api_name}. "
                                f"Available API name list: {KnowledgeGraphAPI.get_valid_api_name_list()}"
                            ),
                        }
                    )
                    return
                # endregion
                # region Get argument list
                # + 1, -1 are used to remove the parentheses
                argument_str = api_str[len(api_name) + 1 : -1]
                raw_argument_list = (
                    KnowledgeGraph._extract_argument_list_from_argument_str(
                        argument_str, list(current_dataset_item.entity_dict.keys())
                    )
                )
                processed_argument_list: list[str | Variable] = []
                for raw_argument in raw_argument_list:
                    processed_argument: str | Variable
                    if raw_argument in current_dataset_item.entity_dict.keys():
                        # region Argument is an entity
                        processed_argument = current_dataset_item.entity_dict[
                            raw_argument
                        ]
                        # endregion
                    else:
                        # region Extract variable index
                        try:
                            variable_index: Optional[int] = (
                                KnowledgeGraph._extract_variable_index_from_argument(
                                    raw_argument
                                )
                            )
                        except:  # noqa
                            session.chat_history.inject(
                                {
                                    "role": Role.USER,
                                    "content": (
                                        f"Cannot extract the variable index from the following API argument: "
                                        f"{raw_argument}"
                                    ),
                                }
                            )
                            return
                        # endregion
                        if variable_index is not None:
                            # region Change variable index to variable
                            try:
                                processed_argument = self.variable_list[variable_index]
                            except:  # noqa
                                error_message = (
                                    self._get_nonexistent_variable_error_message(
                                        variable_index
                                    )
                                )
                                session.chat_history.inject(
                                    {
                                        "role": Role.USER,
                                        "content": error_message,
                                    }
                                )
                                return
                            # endregion
                        else:
                            # region Argument is neither an entity nor a variable
                            processed_argument = raw_argument
                            # endregion
                    processed_argument_list.append(processed_argument)
                # endregion
                # region Get callable API
                api: Callable[..., tuple[Variable | None, str]]
                match api_name:
                    case "get_relations":
                        api = self.knowledge_graph_api.get_relations
                    case "get_neighbors":
                        api = self.knowledge_graph_api.get_neighbors
                    case "intersection":
                        api = self.knowledge_graph_api.intersection
                    case "get_attributes":
                        api = self.knowledge_graph_api.get_attributes
                    case "argmax":
                        api = self.knowledge_graph_api.argmax
                    case "argmin":
                        api = self.knowledge_graph_api.argmin
                    case "count":
                        api = self.knowledge_graph_api.count
                    case _:
                        raise ValueError(f"An API name is not handled: {api_name}")
                # endregion
                # region Check argument count
                api_parameter_list: list[str] = inspect.getfullargspec(api).args
                if api_parameter_list[0] == "self":
                    api_parameter_list = api_parameter_list[1:]
                if len(api_parameter_list) != len(processed_argument_list):
                    # region Construct error_message
                    if len(api_parameter_list) > 1:
                        error_message = f"API {api_name} requires {len(api_parameter_list)} arguments, "
                    else:
                        error_message = f"API {api_name} requires {len(api_parameter_list)} argument, "
                    if len(processed_argument_list) > 1:
                        error_message += f"but {len(processed_argument_list)} arguments are provided."
                    else:
                        error_message += (
                            f"but {len(processed_argument_list)} argument is provided."
                        )
                    # endregion
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": error_message,
                        }
                    )
                    return
                # endregion
                # region Call API with arguments
                try:
                    new_variable, execution_message = api(*processed_argument_list)
                except KnowledgeGraphAPIException as e:
                    error_message = str(e)
                    # region Replace the template in error_message with real value
                    for raw_argument_index, raw_argument in enumerate(
                        raw_argument_list
                    ):
                        error_message = error_message.replace(
                            f"<<ARGUMENT{raw_argument_index}>>", raw_argument
                        )
                    callable_variable_str_list: list[str] = []
                    not_callable_variable_str_list: list[str] = []
                    for variable_index, variable in enumerate(self.variable_list):
                        if variable.is_callable():
                            callable_variable_str_list.append(f"#{variable_index}")
                        else:
                            not_callable_variable_str_list.append(f"#{variable_index}")
                    callable_variable_list_str = (
                        f"[{', '.join(callable_variable_str_list)}]"
                    )
                    not_callable_variable_list_str = (
                        f"[{', '.join(not_callable_variable_str_list)}]"
                    )
                    error_message = error_message.replace(
                        "<<CALLABLE_VARIABLE_LIST_STR>>", callable_variable_list_str
                    )
                    error_message = error_message.replace(
                        "<<NOT_CALLABLE_VARIABLE_LIST_STR>>",
                        not_callable_variable_list_str,
                    )
                    # endregion
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": f"Error in executing '{api_str}'. Error: {error_message}",
                        }
                    )
                    return
                except Exception as e:
                    session.task_output = self._get_default_task_output()
                    raise TaskEnvironmentException(str(e))
                execution_message = execution_message.replace("<<API_STR>>", api_str)
                if "<<NEW_VARIABLE>>" in execution_message:
                    # the execution message contains a variable
                    execution_message = execution_message.replace(
                        "<<NEW_VARIABLE>>", f"#{len(self.variable_list)}"
                    )
                    assert isinstance(new_variable, Variable)  # Type narrowing
                    self.variable_list.append(new_variable)
                session.chat_history.inject(
                    {"role": Role.USER, "content": execution_message}
                )
                return
                # endregion
            case AgentAction.FINISH:
                assert parser_result.content is not None
                try:
                    answer_variable_index = int(parser_result.content)
                except:  # noqa
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": f"Cannot find variable index in final answer.",
                        }
                    )
                    return
                try:
                    answer_variable: Variable = self.variable_list[
                        answer_variable_index
                    ]
                except:  # noqa
                    error_message = self._get_nonexistent_variable_error_message(
                        answer_variable_index
                    )
                    session.chat_history.inject(
                        {
                            "role": Role.USER,
                            "content": error_message,
                        }
                    )
                    return
                try:
                    answer = self.knowledge_graph_api.final_execute(answer_variable)
                except Exception as e:
                    session.task_output = self._get_default_task_output()
                    raise TaskEnvironmentException(str(e))
                session.task_output = {"answer": "<SEP>".join(answer)}
                session.sample_status = SampleStatus.COMPLETED
                return
            case AgentAction.INVALID:
                session.sample_status = SampleStatus.AGENT_VALIDATION_FAILED
                session.finish_reason = parser_result.finish_reason
                session.task_output = self._get_default_task_output()
                return
        # endregion

    def _complete(self, session: Session) -> None:
        # region Preparation
        current_dataset_item: KnowledgeGraphDatasetItem = (
            self._get_current_dataset_item()
        )
        if session.task_output is None:
            # Handle extreme case, such as SampleStatus.TASK_UNKNOWN_ERROR.
            session.task_output = {}
        if session.task_output.get("answer", None) is not None:
            agent_answer_list = str(session.task_output["answer"]).split("<SEP>")
        else:
            agent_answer_list = None
        ground_truth_answer_set = current_dataset_item.answer_set
        # endregion
        # region Calculate metrics
        # region Calculate f1_score
        f1_score: float
        if agent_answer_list is None:
            f1_score = 0
        else:
            agent_answer_set = set(agent_answer_list)
            true_positive = len(ground_truth_answer_set.intersection(agent_answer_set))
            false_positive = len(agent_answer_set - ground_truth_answer_set)
            false_negative = len(ground_truth_answer_set - agent_answer_set)
            if true_positive == 0:
                f1_score = 0
            else:
                precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative)
                f1_score = 2 * precision * recall / (precision + recall)
        # endregion
        # region Calculate exact_match
        if agent_answer_list is None:
            exact_match = False
        else:
            exact_match = ground_truth_answer_set == set(agent_answer_list)
        # endregion
        # endregion
        # region Record evaluation results
        session.evaluation_record.outcome = SessionEvaluationOutcome.from_bool(
            exact_match
        )
        session.evaluation_record.detail_dict = {
            "f1_score": f1_score,
            "executable_flag": session.task_output.get("answer", None) is not None,
            # Do not set exact_match here, because it is already included in the outcome
        }
        # endregion
        # region Clean up
        self.variable_list = None
        # endregion

    def _release(self) -> None:
        return  # Do nothing

    def calculate_metric(
        self, session_partial_list: Sequence[SessionMetricCalculationPartial]
    ) -> MetricDict:
        # region Calculate general metrics
        skill_metric_dict = self._calculate_metric_based_on_skill(
            KnowledgeGraphSkillUtility, session_partial_list
        )
        difficulty_level_metric_dict = self._calculate_metric_based_on_difficulty_level(
            session_partial_list
        )
        overall_metric_dict = Task._calculate_overall_metric(session_partial_list)
        # endregion
        # region Calculate task-specific metrics
        f1_score_numerator: float = 0
        executable_rate_numerator: int = 0
        for session_partial in session_partial_list:
            if session_partial.evaluation_record.detail_dict is not None:
                f1_score = session_partial.evaluation_record.detail_dict.get(
                    "f1_score", 0.0
                )
                if isinstance(
                    f1_score, (int, float)
                ):  # The if statement is used for type narrowing
                    f1_score_numerator += float(f1_score)
                executable_flag = session_partial.evaluation_record.detail_dict.get(
                    "executable_flag", False
                )
                if isinstance(
                    executable_flag, bool
                ):  # The if statement is used for type narrowing
                    executable_rate_numerator += int(executable_flag)
        additional_metric_dict = {
            "f1_score": f1_score_numerator / len(session_partial_list),
            "executable_rate": executable_rate_numerator / len(session_partial_list),
        }
        overall_metric_dict["additional"] = additional_metric_dict
        # endregion
        metric_dict = {
            "skill": skill_metric_dict,
            "difficulty_level": difficulty_level_metric_dict,
            "overall": overall_metric_dict,
        }
        return metric_dict
