import json
import re
import math
from typing import Optional, Self, Mapping, Any, Sequence
from pydantic import BaseModel, model_validator
from enum import StrEnum, unique

from src.tasks.task import (
    Task,
    DatasetItem,
    SkillUtility,
    AgentResponseParserResult,
    AgentAction,
)
from .container import DBBenchContainer
from src.typings import (
    SampleIndex,
    SampleStatus,
    TaskEnvironmentException,
    TaskReleaseException,
    Session,
    TaskName,
    Role,
    SessionEvaluationOutcome,
    MetricDict,
    SessionMetricCalculationPartial,
)
from src.factories.chat_history_item import ChatHistoryItemFactory


class DBBenchSkillUtility(SkillUtility):
    _SKILL_TO_LEVEL_DICT = {
        key: 0
        for key in sorted(
            [
                "select",
                "insert",
                "delete",
                "update",
                "where_single_condition",
                "where_multiple_conditions",
                "order_by_single_column",
                "limit_only",
                "column_alias",
                "table_alias",
                "where_nested_conditions",
                "group_by_single_column",
                "group_by_multiple_columns",
                "having_single_condition_with_aggregate",
                "having_multiple_conditions_with_aggregate",
                "having_aggregate_calculation",
                "order_by_multiple_columns_same_direction",
                "order_by_multiple_columns_different_directions",
                "limit_and_offset",
                "subquery_single",
                "subquery_multiple",
                "subquery_nested",
            ]
        )
    }


@unique
class AnswerType(StrEnum):
    MD5 = "md5"
    DIRECT = "direct"


class DBBenchType:
    RowValue = int | str | float
    Row = tuple[RowValue, ...]
    AnswerDirect = Sequence["DBBenchType.Row"]


class AnswerInfo(BaseModel):
    answer_type: AnswerType
    answer_md5: Optional[str]
    answer_direct: Optional[DBBenchType.AnswerDirect]
    ground_truth_sql: str  # Only used for validation

    @model_validator(mode="after")
    def check_answer_type(self) -> Self:
        assert self.answer_direct is not None or self.answer_md5 is not None
        assert not (self.answer_direct is not None and self.answer_md5 is not None)
        answer_type_assigned = self.answer_type
        answer_type_from_field = (
            AnswerType.MD5 if self.answer_md5 else AnswerType.DIRECT
        )
        assert answer_type_assigned == answer_type_from_field
        sql_type = self.ground_truth_sql.split()[0].upper()
        if sql_type == "SELECT":
            answer_type_from_sql = AnswerType.DIRECT
        else:
            assert sql_type in ("INSERT", "DELETE", "UPDATE")
            answer_type_from_sql = AnswerType.MD5
        assert answer_type_assigned == answer_type_from_sql
        return self


class ColumnInfo(BaseModel):
    name: str
    type: str


class TableInfo(BaseModel):
    name: str
    row_list: Sequence[DBBenchType.Row]
    column_info_list: Sequence[ColumnInfo]


class DBBenchDatasetItem(DatasetItem):
    instruction: str
    answer_info: AnswerInfo
    database_name: str
    table_info: TableInfo
    skill_list: list[str]

    def get_skill_list(self) -> list[str]:
        return self.skill_list

    def get_difficulty_level(self) -> int:
        return 0


class DirectTypeAnswerValidator:
    @staticmethod
    def _is_two_float_equal(float1: float, float2: float) -> bool:
        return math.isclose(float1, float2, rel_tol=1e-06, abs_tol=1e-06)

    @staticmethod
    def _get_tuple_position_list(answer: str) -> Optional[list[tuple[int, int]]]:
        # Return None if parsing fails
        tuple_position_list = []
        start_index: Optional[int] = None
        end_index: Optional[int] = None
        disable_flag = False
        disable_word = "Decimal"
        for index, char in enumerate(answer):
            if char == "(":
                if disable_word == answer[index - len(disable_word) : index]:
                    if disable_flag:
                        return None
                    disable_flag = True
                    continue
                else:
                    start_index = index
            elif char == ")":
                if disable_flag:
                    disable_flag = False
                    continue
                end_index = index
                if start_index is None:
                    return None
                tuple_position_list.append((start_index, end_index))
                start_index = None
                end_index = None
            else:
                pass
                # Can add further validation here
        if start_index is not None or end_index is not None:
            return None
        return tuple_position_list

    @staticmethod
    def _validate_single_tuple_str(
        agent_tuple_str: str, ground_truth_tuple: DBBenchType.Row
    ) -> bool:
        # region Get agent_answer_element_list
        raw_agent_answer_element_list = re.split(r"\s*,\s*", agent_tuple_str)
        agent_answer_element_list = []
        for element in raw_agent_answer_element_list:
            if not element.strip() == "":
                agent_answer_element_list.append(element.strip())
        # endregion
        # region Validate the length
        if len(agent_answer_element_list) != len(ground_truth_tuple):
            return False
        # endregion
        # region Validate a single element
        for agent_answer_element, ground_truth_element in zip(
            agent_answer_element_list, ground_truth_tuple
        ):
            if isinstance(ground_truth_element, (int, float)):
                # region Handle int and float
                agent_answer_element = re.sub(
                    r'^Decimal\((["\']?)(.*?)(["\']?)\)$', r"\2", agent_answer_element
                )
                # region Handle case """'-55'"""
                start_index = 0
                valid_char_in_float_str = "0123456789.-"
                for i in range(len(agent_answer_element)):
                    if agent_answer_element[i] in valid_char_in_float_str:
                        break
                    start_index = i
                end_index = len(agent_answer_element)
                for i in range(len(agent_answer_element) - 1, -1, -1):
                    if agent_answer_element[i] in valid_char_in_float_str:
                        break
                    end_index = i + 1
                # endregion
                try:
                    processed_element = float(
                        agent_answer_element[start_index:end_index]
                    )
                except:  # noqa
                    return False
                if not DirectTypeAnswerValidator._is_two_float_equal(
                    processed_element, ground_truth_element
                ):
                    return False
                # endregion
            elif isinstance(ground_truth_element, str):
                # region Handle str
                if 0 < len(agent_answer_element) - len(ground_truth_element) <= 2:
                    head_strip_flag = False
                    tail_strip_flag = False
                    for char in ["'", '"']:
                        if (
                            agent_answer_element.startswith(char)
                            and not ground_truth_element.startswith(char)
                            and not head_strip_flag
                        ):
                            agent_answer_element = agent_answer_element[1:]
                            head_strip_flag = True
                        if (
                            agent_answer_element.endswith(char)
                            and not ground_truth_element.endswith(char)
                            and not tail_strip_flag
                        ):
                            agent_answer_element = agent_answer_element[:-1]
                            tail_strip_flag = True
                    if agent_answer_element != ground_truth_element:
                        return False
                if agent_answer_element != ground_truth_element:
                    return False
                # endregion
            else:
                # This could never reach if DBBenchType.Row = list[tuple[int | str | float, ...]]
                # It is only kept for the correctness of the code
                return False
        # endregion
        return True

    @staticmethod
    def validate(agent_answer: str, ground_truth: DBBenchType.AnswerDirect) -> bool:
        if (
            tuple_position_list := DirectTypeAnswerValidator._get_tuple_position_list(
                agent_answer
            )
        ) is None:
            return False
        if len(tuple_position_list) == 0:
            if len(ground_truth) == 0:
                return True
            if len(ground_truth) != 1:
                return False
            return DirectTypeAnswerValidator._validate_single_tuple_str(
                agent_answer, ground_truth[0]
            )
        if len(tuple_position_list) != len(ground_truth):
            return False
        for tuple_position, ground_truth_tuple in zip(
            tuple_position_list, ground_truth
        ):
            start_index, end_index = tuple_position
            tuple_str = agent_answer[start_index + 1 : end_index]
            if not DirectTypeAnswerValidator._validate_single_tuple_str(
                tuple_str, ground_truth_tuple
            ):
                return False
        return True


class DBBench(Task[DBBenchDatasetItem]):
    def __init__(
        self,
        task_name: TaskName,
        chat_history_item_factory: ChatHistoryItemFactory,
        data_file_path: str,
        max_round: int,
    ):
        super().__init__(task_name, chat_history_item_factory, max_round)
        data = json.load(open(data_file_path))
        dataset: dict[SampleIndex, DBBenchDatasetItem] = {}
        for key, entry in data.items():
            dataset_item = DBBench._construct_dataset_item(entry)
            dataset[key] = dataset_item
        self._set_dataset(dataset)
        # Construct docker container immediately
        self.container = DBBenchContainer()

    @staticmethod
    def _construct_dataset_item(entry: dict[str, Any]) -> DBBenchDatasetItem:
        # region Construct answer_info
        answer_md5: Optional[str] = entry["answer_info"]["md5"]
        raw_answer_direct = entry["answer_info"]["direct"]
        answer_direct: Optional[list[DBBenchType.Row]]
        if raw_answer_direct is not None:
            answer_direct = []
            for answer_item in raw_answer_direct:
                assert isinstance(answer_item, list)
                answer_direct.append(tuple(answer_item))
        else:
            answer_direct = None
        if answer_md5 is not None:
            answer_type = AnswerType.MD5
        else:
            answer_type = AnswerType.DIRECT
        ground_truth_sql = entry["answer_info"]["sql"].strip()
        answer_info = AnswerInfo(
            answer_type=answer_type,
            answer_md5=answer_md5,
            answer_direct=answer_direct,
            ground_truth_sql=ground_truth_sql,
        )
        # endregion
        # region Get database_name
        # The name of database is set to the same as the name of table
        database_name = entry["table_info"]["name"]
        # endregion
        # region Get table_info
        name = entry["table_info"]["name"]
        row_list = entry["table_info"]["row_list"]
        column_info_list: list[ColumnInfo] = []
        for column in entry["table_info"]["column_info_list"]:
            column_info_list.append(ColumnInfo(**column))
        table_info = TableInfo(
            name=name, row_list=row_list, column_info_list=column_info_list
        )
        # endregion
        # region Get skill_list
        skill_list = entry["skill_list"]
        assert all([DBBenchSkillUtility.is_valid_skill(skill) for skill in skill_list])
        # endregion
        # region Get instruction
        question_prefix = entry["instruction"]
        question_suffix = (
            f"The name of this table is {table_info.name}, and the headers of this table are "
            f"{', '.join([column_info.name for column_info in column_info_list])}."
        )
        instruction = f"{question_prefix}\n{question_suffix}"
        # endregion
        # region Construct DatasetItem
        dataset_item = DBBenchDatasetItem(
            instruction=instruction,
            answer_info=answer_info,
            database_name=database_name,
            table_info=table_info,
            skill_list=skill_list,
        )
        # endregion
        return dataset_item

    @staticmethod
    def _build_init_sql(dataset_item: DBBenchDatasetItem) -> str:
        table_info = dataset_item.table_info
        # region Construct column_str and column_name_str
        column_info_list = table_info.column_info_list
        column_str = ",".join(
            [
                f"`{column_info.name}` {column_info.type}"
                for column_info in column_info_list
            ]
        )
        column_name_str = ",".join(
            [f"`{column_info.name}`" for column_info in column_info_list]
        )
        # endregion
        # region Construct item_str
        row_list = table_info.row_list
        item_list = []
        item_value_list = []
        for row in row_list:
            item = "("
            for value in row:
                item += "'%s',"
                if isinstance(value, str):
                    value = value.replace("'", "''")
                item_value_list.append(value)
            item = item[:-1] + ")"  # "-1" is to remove the last ","
            item_list.append(item)
        item_str = ",".join(item_list)
        item_str = item_str % tuple(item_value_list)
        # endregion
        # region Construct the final sql
        table_name = table_info.name
        database_name = dataset_item.database_name
        sql = (
            f"CREATE DATABASE IF NOT EXISTS `{database_name}`;\n"  # noqa
            f"USE `{database_name}`;\n"
            f"CREATE TABLE IF NOT EXISTS `{table_name}` ({column_str});\n"
            f"INSERT INTO `{table_name}` ({column_name_str}) VALUES {item_str};\n"
            f"COMMIT;\n"
        )
        # endregion
        return sql

    def _get_task_output(self, answer: str) -> dict[str, str]:
        dataset_item: DBBenchDatasetItem = self._get_current_dataset_item()
        answer_info = dataset_item.answer_info
        match answer_info.answer_type:
            case AnswerType.MD5:
                column_info_list = dataset_item.table_info.column_info_list
                column_name_str = ",".join(
                    [f"`{column_info.name}`" for column_info in column_info_list]
                )
                database_name = dataset_item.database_name
                table_name = dataset_item.table_info.name
                md5_query = (
                    f"select md5(group_concat(rowhash order by rowhash)) as hash "  # noqa
                    f"from( SELECT substring(MD5(CONCAT_WS(',', {column_name_str})), 1, 5) AS rowhash "
                    f"FROM `{table_name}`) as sub;"
                )
                try:
                    answer = self.container.execute(md5_query, database_name)
                    # "[('67b8112b11e457cbd639f01ce6078a06',)]" -> "67b8112b11e457cbd639f01ce6078a06"
                    # "[(None,)]" -> "None"
                    answer_match = re.search(r"\('?(.*?)'?,\)", answer)
                    assert answer_match is not None
                    answer = answer_match.group(1)
                except Exception as e:
                    raise TaskEnvironmentException(str(e))
                return {"answer": answer}
            case AnswerType.DIRECT:
                return {"answer": answer}
            case _:
                raise NotImplementedError()

    def _get_default_task_output(self) -> dict[str, Optional[str]]:
        task_output: Mapping[str, str | None] = self._get_task_output("")
        return dict(task_output)

    @staticmethod
    def _parse_agent_response(agent_response: str) -> AgentResponseParserResult:
        if (
            agent_action_match := re.search(
                r"Action: (Operation|Answer)", agent_response
            )
        ) is None:
            return AgentResponseParserResult(
                action=AgentAction.INVALID,
                content=None,
                finish_reason=r'Can not find action in agent response. Pattern: "Action: (Operation|Answer)\n',
            )
        match agent_action_match.group(1):
            case "Operation":
                if (
                    sql_match := re.search(r"```sql\n([\s\S]*?)\n```", agent_response)
                ) is None:
                    return AgentResponseParserResult(
                        action=AgentAction.INVALID,
                        content=None,
                        finish_reason=r'Can not find SQL in agent response. Pattern: "```sql\n([\s\S]*?)\n```',
                    )
                sql = sql_match.group(1).strip().replace("\n", " ")
                return AgentResponseParserResult(
                    action=AgentAction.EXECUTE, content=sql, finish_reason=None
                )
            case "Answer":
                if (
                    answer_match := re.search(r"\nFinal Answer:(.*)", agent_response)
                ) is None:
                    return AgentResponseParserResult(
                        action=AgentAction.INVALID,
                        content=None,
                        finish_reason=(
                            r"Can not find final answer in agent response. "
                            r"Pattern: \nFinal Answer:(.*)"
                        ),
                    )
                answer = answer_match.group(1).strip()
                return AgentResponseParserResult(
                    action=AgentAction.FINISH, content=answer, finish_reason=None
                )
            case _:
                raise RuntimeError(
                    "An unexpected action is matched, Check the code and fix the bug."
                )

    def _reset(self, session: Session) -> None:
        # Initialize the database and chat history
        current_dataset_item: DBBenchDatasetItem = self._get_current_dataset_item()
        init_sql = DBBench._build_init_sql(current_dataset_item)
        self.container.execute(init_sql)
        session.chat_history.inject(
            self.chat_history_item_factory.construct(0, expected_role=Role.USER)
        )
        session.chat_history.inject(
            self.chat_history_item_factory.construct(1, expected_role=Role.AGENT)
        )
        prompt = current_dataset_item.instruction
        session.chat_history.inject({"role": Role.USER, "content": prompt})

    def _interact(self, session: Session) -> None:
        # region Preparation
        parser_result = DBBench._parse_agent_response(
            session.chat_history.get_item_deep_copy(-1).content
        )
        current_dataset_item: DBBenchDatasetItem = self._get_current_dataset_item()
        # endregion
        # region Execute action
        match parser_result.action:
            case AgentAction.EXECUTE:
                sql = parser_result.content
                assert sql is not None, "Check DBBench._parse_agent_response()."
                database_name = current_dataset_item.database_name
                try:
                    user_response = self.container.execute(sql, database_name)
                except Exception as e:
                    session.task_output = self._get_default_task_output()
                    raise TaskEnvironmentException(str(e))
                session.chat_history.inject(
                    {"role": Role.USER, "content": user_response}
                )
                return
            case AgentAction.FINISH:
                answer = parser_result.content
                assert answer is not None, "Check DBBench._parse_agent_response()."
                try:
                    task_output: Mapping[str, str | None] = self._get_task_output(
                        answer
                    )
                except Exception as e:
                    raise TaskEnvironmentException(str(e))
                session.task_output = dict(task_output)
                session.sample_status = SampleStatus.COMPLETED
                return
            case AgentAction.INVALID:
                session.sample_status = SampleStatus.AGENT_VALIDATION_FAILED
                session.finish_reason = parser_result.finish_reason
                try:
                    session.task_output = self._get_default_task_output()
                except Exception as e:
                    session.finish_reason = (
                        f"Two errors occurred: "
                        f"[Error 1]: {session.finish_reason} "
                        f"[Error 2]: Error happens in getting task output. Detail: {str(e)}"
                    )
                return
            case _:
                raise TypeError()
        # endregion

    def _complete(self, session: Session) -> None:
        # region Clean up
        current_dataset_item: DBBenchDatasetItem = self._get_current_dataset_item()
        self.container.execute(
            f"drop database `{current_dataset_item.database_name}`"  # noqa
        )
        # endregion
        # region Validate the correctness of agent answer
        # region Get agent_answer and answer_info
        try:
            assert isinstance(session.task_output, dict)
            agent_answer = session.task_output["answer"]
            assert isinstance(agent_answer, str)
        except:  # noqa
            # Usually because SampleStatus.TASK_UNKNOWN_ERROR.
            # Do not do further justification here, since it is too extreme.
            agent_answer = ""
        answer_info = current_dataset_item.answer_info
        # endregion
        # region Validate answer
        match answer_info.answer_type:
            case AnswerType.MD5:
                correct_flag = agent_answer == answer_info.answer_md5
            case AnswerType.DIRECT:
                ground_truth = answer_info.answer_direct
                assert (
                    ground_truth is not None
                ), "Check DBBench._construct_dataset_item()."
                correct_flag = DirectTypeAnswerValidator.validate(
                    agent_answer, ground_truth
                )
            case _:
                raise NotImplementedError()
        # endregion
        # endregion
        session.evaluation_record.outcome = SessionEvaluationOutcome.from_bool(
            correct_flag
        )

    def _release(self) -> None:
        try:
            self.container.delete()
        except Exception as e:
            raise TaskReleaseException(str(e))

    def calculate_metric(
        self, session_partial_list: Sequence[SessionMetricCalculationPartial]
    ) -> MetricDict:
        skill_metric_dict = self._calculate_metric_based_on_skill(
            DBBenchSkillUtility, session_partial_list
        )
        difficulty_level_metric_dict = self._calculate_metric_based_on_difficulty_level(
            session_partial_list
        )
        overall_metric_dict = Task._calculate_overall_metric(session_partial_list)
        metric_dict = {
            "skill": skill_metric_dict,
            "difficulty_level": difficulty_level_metric_dict,
            "overall": overall_metric_dict,
        }
        return metric_dict
