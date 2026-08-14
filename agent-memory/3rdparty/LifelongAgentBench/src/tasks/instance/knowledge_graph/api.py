import json
import re
from typing import Union, Optional, Sequence, Any
import os
from pydantic import BaseModel
import inspect
from enum import StrEnum

from .utils.logic_form_util import LogicFormUtil
from .utils.sparql_executor import SparqlExecutor


class Variable(BaseModel):
    type: str
    program: str

    def __hash__(self) -> int:
        return hash(self.program)

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Variable):
            return self.program == __value.program
        else:
            return False

    def __repr__(self) -> str:
        return self.program

    def is_callable(self) -> bool:
        for not_callable_name in [
            "count",
            "argmax",
            "argmin",
        ]:
            if self.program.startswith(f"({not_callable_name.upper()} "):
                return False
        return True


class KnowledgeGraphAPIException(Exception):
    pass


class KnowledgeGraphAPI:
    class ExtremumFunction(StrEnum):
        ARGMAX = "argmax"
        ARGMIN = "argmin"

    def __init__(self, ontology_dir_path: str, sparql_executor: SparqlExecutor):
        with open(os.path.join(ontology_dir_path, "vocab.json")) as f:
            vocab = json.load(f)
            self.attributes = vocab["attributes"]
            self.relations = vocab["relations"]
        self.range_info = {}
        with open(os.path.join(ontology_dir_path, "fb_roles"), "r") as f:
            for line in f:
                line = line.replace("\n", "")
                fields = line.split(" ")
                self.range_info[fields[1]] = fields[2]
        self.variable_to_relations_cache: dict[Variable | str, list[str]] = {}
        self.variable_to_attributes_cache: dict[Variable, list[str]] = {}
        self.sparql_executor = sparql_executor

    @staticmethod
    def _ensure_variable(caller_name: str, argument_list: Sequence[Any]) -> None:
        error_message: Optional[str] = None
        for argument_index, argument in enumerate(argument_list):
            if not isinstance(argument, Variable):
                if error_message is None:
                    error_message = f"{caller_name}: "
                else:
                    error_message += ", "
                error_message += f"<<ARGUMENT{argument_index}>> is not a Variable"
        if error_message is not None:
            raise KnowledgeGraphAPIException(error_message)

    def _validate_attribute(
        self, caller_name: str, variable: Variable, attribute: str
    ) -> None:
        if variable not in self.variable_to_attributes_cache.keys():
            raise KnowledgeGraphAPIException(
                f"{caller_name}: "
                f"Use {self.get_attributes.__name__} to get attributes of the Variable <<ARGUMENT0>> first"
            )
        if attribute not in self.variable_to_attributes_cache[variable]:
            raise KnowledgeGraphAPIException(
                f"{caller_name}: <<ARGUMENT1>> is not an attribute of the Variable <<ARGUMENT0>>. "
                f"The attributes of the Variable <<ARGUMENT0>> are: {self.variable_to_attributes_cache[variable]}"
            )

    @staticmethod
    def _validate_variable(caller_name: str, variable_list: Sequence[Variable]) -> None:
        """
        Variable returned by argmax, argmin and count can only be used as final answer.
        """
        error_message: Optional[str] = None
        for variable_index, variable in enumerate(variable_list):
            if variable.is_callable():
                continue
            if error_message is None:
                error_message = f"{caller_name}: "
            else:
                error_message += " "
            # (COUNT (...)) -> COUNT -> count
            # (ARGMAX (...) attribute) -> ARGMAX -> argmax
            api_name = variable.program.split(" ")[0][1:].lower()
            error_message += (
                f"<<ARGUMENT{variable_index}>> is a Variable returned by {api_name}, "
                f"it can only be used as final answer."
            )
        if error_message is not None:
            error_message += (
                f" Remember, Variables (<<CALLABLE_VARIABLE_LIST_STR>>) returned by get_relations, get_neighbors, intersection, get_attributes "
                f"can be used as inputs for subsequent actions or as final answers, "
                f"and Variables (<<NOT_CALLABLE_VARIABLE_LIST_STR>>) returned by get_relations, get_neighbors, intersection, get_attributes "
                f"can only be used as final answers."
            )
            raise KnowledgeGraphAPIException(error_message)

    def reset_cache(self) -> None:
        self.variable_to_relations_cache = {}
        self.variable_to_attributes_cache = {}

    @staticmethod
    def _construct_execution_message(observation: str) -> str:
        return f"<<API_STR>> executes successfully. Observation: {observation}"

    @staticmethod
    def _is_valid_entity(argument: str) -> bool:
        # According to https://www.wikidata.org/wiki/Property:P646
        # Freebase ID (or entity ID) can only start with 'g' or 'm', instead of 'm' or 'f'.
        if re.match(r"^([gm])\.[\w_]+$", argument):
            return True
        else:
            return False

    def final_execute(self, variable: Variable) -> list[str]:
        program = variable.program
        processed_code = LogicFormUtil.postprocess_raw_code(program)
        sparql_query = LogicFormUtil.lisp_to_sparql(processed_code)
        results = self.sparql_executor.execute_query(sparql_query)
        return results

    def get_relations(self, argument: Union[Variable, str]) -> tuple[None, str]:
        # region Validate argument
        if isinstance(argument, Variable):
            KnowledgeGraphAPI._validate_variable("get_relations", [argument])
        elif KnowledgeGraphAPI._is_valid_entity(argument):
            pass
        else:
            raise KnowledgeGraphAPIException(
                "get_relations: <<ARGUMENT0>> is neither a Variable nor an entity. "
                "The argument of get_relations must be a Variable or an entity."
            )
        # endregion
        if isinstance(argument, Variable):
            program = argument.program
            processed_code = LogicFormUtil.postprocess_raw_code(program)
            sparql_query = LogicFormUtil.lisp_to_sparql(processed_code)
            clauses = sparql_query.split("\n")
            new_clauses = [
                clauses[0],
                "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{",
            ]
            new_clauses.extend(clauses[1:])
            new_clauses.append("}\n}")
            new_query = "\n".join(new_clauses)
            out_relations = self.sparql_executor.execute_query(new_query)
        else:
            out_relations = self.sparql_executor.get_out_relations(argument)
        out_relations = sorted(
            list(set(out_relations).intersection(set(self.relations)))
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"[{', '.join(out_relations)}]"
        )
        self.variable_to_relations_cache[argument] = out_relations
        return None, execution_message

    def get_neighbors(
        self, argument: Union[Variable, str], relation: str
    ) -> tuple[Variable, str]:
        # region Validate arguments
        if isinstance(argument, Variable):
            KnowledgeGraphAPI._validate_variable("get_neighbors", [argument])
        elif KnowledgeGraphAPI._is_valid_entity(argument):
            pass
        else:
            raise KnowledgeGraphAPIException(
                "get_neighbors: <<ARGUMENT0>> is neither a Variable nor an entity. "
                "The first argument of get_neighbors must be a Variable or an entity."
            )
        if argument not in self.variable_to_relations_cache.keys():
            raise KnowledgeGraphAPIException(
                f"get_neighbors: Execute get_relations for <<ARGUMENT0>> before executing get_neighbors"
            )
        if relation not in self.variable_to_relations_cache[argument]:
            raise KnowledgeGraphAPIException(
                f"get_neighbors: <<ARGUMENT1>> is not a relation of the <<ARGUMENT0>>. "
                f"<<ARGUMENT0>> has the following relations: {self.variable_to_relations_cache[argument]}"
            )
        # endregion
        new_variable = Variable(
            type=self.range_info[relation],
            program=f"(JOIN {relation + '_inv'} {argument.program if isinstance(argument, Variable) else argument})",
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {self.range_info[relation]}"
        )
        return new_variable, execution_message

    @staticmethod
    def intersection(variable1: Variable, variable2: Variable) -> tuple[Variable, str]:
        # region Validate arguments
        caller_name = "intersection"
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable1, variable2])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable1, variable2])
        if variable1.type != variable2.type:
            raise KnowledgeGraphAPIException(
                "intersection: Two Variables must have the same type"
            )
        # endregion
        new_variable = Variable(
            type=variable1.type,
            program=f"(AND {variable1.program} {variable2.program})",
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {variable1.type}"
        )
        return new_variable, execution_message

    @staticmethod
    def union(variable1: Variable, variable2: Variable) -> tuple[Variable, str]:
        # region Validate arguments
        # The function is not included in the prompt
        KnowledgeGraphAPI._ensure_variable("union", [variable1, variable2])

        if variable1.type != variable2.type:
            raise KnowledgeGraphAPIException(
                "union: Two Variables must have the same type"
            )
        # endregion
        new_variable = Variable(
            type=variable1.type, program=f"(OR {variable1.program} {variable2.program})"
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {variable1.type}"
        )
        return new_variable, execution_message

    @staticmethod
    def count(variable: Variable) -> tuple[Variable, str]:
        # region Validate arguments
        caller_name = "count"
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable])
        # endregion
        new_variable = Variable(type="type.int", program=f"(COUNT {variable.program})")
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which is a number"
        )
        return new_variable, execution_message

    def get_attributes(self, variable: Variable) -> tuple[None, str]:
        # region Validate variable
        caller_name = "get_attributes"
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable])
        # endregion
        program = variable.program
        processed_code = LogicFormUtil.postprocess_raw_code(program)
        sparql_query = LogicFormUtil.lisp_to_sparql(processed_code)
        clauses = sparql_query.split("\n")
        new_clauses = [clauses[0], "SELECT DISTINCT ?rel\nWHERE {\n?x ?rel ?obj .\n{"]
        new_clauses.extend(clauses[1:])
        new_clauses.append("}\n}")
        new_query = "\n".join(new_clauses)
        out_relations = self.sparql_executor.execute_query(new_query)
        out_relations = sorted(
            list(set(out_relations).intersection(set(self.attributes)))
        )
        self.variable_to_attributes_cache[variable] = out_relations
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"[{', '.join(out_relations)}]"
        )
        return None, execution_message

    def _find_extremum_by_attribute(
        self,
        variable: Variable,
        attribute: str,
        extremum_function: "KnowledgeGraphAPI.ExtremumFunction",
    ) -> tuple[Variable, str]:
        # region Validate arguments
        caller_name = inspect.stack()[1].function
        KnowledgeGraphAPI._ensure_variable(caller_name, [variable])
        KnowledgeGraphAPI._validate_variable(caller_name, [variable])
        self._validate_attribute(caller_name, variable, attribute)
        # endregion
        match extremum_function:
            case KnowledgeGraphAPI.ExtremumFunction.ARGMAX:
                function_name = "ARGMAX"
            case KnowledgeGraphAPI.ExtremumFunction.ARGMIN:
                function_name = "ARGMIN"
            case _:
                raise ValueError("This cannot happen.")
        new_variable = Variable(
            type=variable.type,
            program=f"({function_name} {variable.program} {attribute})",
        )
        execution_message = KnowledgeGraphAPI._construct_execution_message(
            f"Variable <<NEW_VARIABLE>>, which are instances of {variable.type}"
        )
        return new_variable, execution_message

    def argmax(self, variable: Variable, attribute: str) -> tuple[Variable, str]:
        return self._find_extremum_by_attribute(
            variable, attribute, KnowledgeGraphAPI.ExtremumFunction.ARGMAX
        )

    def argmin(self, variable: Variable, attribute: str) -> tuple[Variable, str]:
        return self._find_extremum_by_attribute(
            variable, attribute, KnowledgeGraphAPI.ExtremumFunction.ARGMIN
        )

    @staticmethod
    def get_valid_api_name_list() -> list[str]:
        return [
            # "final_execute",
            "get_relations",
            "get_neighbors",
            "intersection",
            # "union",
            "count",
            "get_attributes",
            "argmax",
            "argmin",
        ]
