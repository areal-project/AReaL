from pydantic import BaseModel, field_validator, ValidationError
from typing import Literal, Any, Optional, TypeGuard, get_args
import builtins
import re
from enum import Enum, StrEnum, unique
from abc import ABC, abstractmethod


@unique
class InstanceFactoryType(StrEnum):
    SimpleImmutableType = "simple_immutable_type"
    Enum = "enum"
    General = "general"


class InstanceFactoryInterface(ABC):
    @abstractmethod
    def create(self) -> Any:
        raise NotImplementedError()


class SimpleImmutableTypeInstanceFactoryUtility:
    ValidType = Literal[
        "bool",
        "NoneType",
        "int",
        "float",
        "str",
    ]

    @classmethod
    def is_simple_immutable_type(
        cls, value_type_str: str
    ) -> TypeGuard["SimpleImmutableTypeInstanceFactoryUtility.ValidType"]:
        return value_type_str in get_args(cls.ValidType)


class SimpleImmutableTypeInstanceFactory(BaseModel, InstanceFactoryInterface):
    # Do not modify the supported types.
    # Cannot support mutable types, such as list, dict, set, etc.
    # Cannot support iterable types, such as tuple, frozenset, etc.

    type: SimpleImmutableTypeInstanceFactoryUtility.ValidType
    value: str

    def create(self) -> Any:
        # region Handle special cases
        if self.type == "bool":
            return self.value == "True"
        if self.type == "NoneType":
            return None
        # endregion
        # region Handle common cases
        if self.type in {"int", "float", "str"}:
            class_type = getattr(builtins, self.type)
            return class_type(self.value)
        # endregion
        raise NotImplementedError(
            "For other built-in types, Wrapped it by pydantic.BaseModel."
        )


class EnumInstanceFactory(BaseModel, InstanceFactoryInterface):
    module: str
    value: str | int

    def create(self) -> Any:
        return GeneralInstanceFactory(
            module=self.module,
            parameters={"value": self.value},
        ).create()


class GeneralInstanceFactory(BaseModel, InstanceFactoryInterface):
    # This class may better be moved to utils directory.
    # But I keep it here to avoid circular import.
    module: str
    parameters: dict[str, Any] = {}

    @field_validator("parameters", mode="before")  # noqa
    @classmethod
    def _ensure_dict(cls, v: Optional[dict[str, Any]]) -> dict[str, Any]:
        # Follow the documentation, this function is defined as a class method
        # https://docs.pydantic.dev/2.10/concepts/validators/#using-the-decorator-pattern
        if v is None:
            return {}
        return v

    def create(self) -> Any:
        # print('>>>>>>>> ', self.module, self.parameters)
        splits = self.module.split(".")
        if len(splits) == 0:
            raise Exception("Invalid module name: {}".format(self.module))
        for parameter in self.parameters:
            try:
                self.parameters[parameter] = self.model_validate(
                    self.parameters[parameter]
                ).create()
            except ValidationError:
                pass
        if len(splits) == 1:
            g = globals()
            if self.module in g:
                class_type = g[self.module]
            else:
                # class_type = getattr(builtins, self.module)
                raise RuntimeError(
                    "GeneralInstanceFactory cannot instantiate built-in types perfectly. "
                    "Use SimpleImmutableTypeInstanceFactory instead."
                )
            return class_type(**self.parameters)
        else:
            path = ".".join(self.module.split(".")[:-1])
            mod = __import__(path, fromlist=[self.module.split(".")[-1]])
            return getattr(mod, self.module.split(".")[-1])(**self.parameters)


class InstanceFactoryUtility:
    @staticmethod
    def _get_type_str(value_type: type) -> str:
        pattern = (
            r"<class '([^']+)'>"  # <class 'int'> -> int, <class 'src.Foo'> -> src.Foo
        )
        if type_str_match := re.search(pattern, str(value_type)):
            return type_str_match.group(1)
        else:
            raise ValueError(f"Cannot get type string from {value_type}")

    @staticmethod
    def create_instance_factory_for_http_transfer(
        value: Any,
    ) -> tuple[
        SimpleImmutableTypeInstanceFactory
        | EnumInstanceFactory
        | GeneralInstanceFactory,
        InstanceFactoryType,
    ]:  # (an instance of instance_factory, instance_factory_type)
        if type(value).__module__ == "builtins":
            value_type_str = type(value).__name__
            assert SimpleImmutableTypeInstanceFactoryUtility.is_simple_immutable_type(
                value_type_str
            )
            return (
                SimpleImmutableTypeInstanceFactory(
                    type=value_type_str,
                    value=str(value),
                ),
                InstanceFactoryType.SimpleImmutableType,
            )
        elif isinstance(value, Enum):
            module = f"{value.__class__.__module__}.{value.__class__.__name__}"
            enum_value = value.value
            assert isinstance(enum_value, (str, int))
            return (
                EnumInstanceFactory(
                    module=module,
                    value=enum_value,
                ),
                InstanceFactoryType.Enum,
            )
        elif isinstance(value, BaseModel):
            value_type_str = InstanceFactoryUtility._get_type_str(type(value))
            return (
                GeneralInstanceFactory(
                    module=value_type_str,
                    parameters=value.model_dump(),
                ),
                InstanceFactoryType.General,
            )
        else:
            raise NotImplementedError(
                "For other types, You can "
                "1. Make it a subclass of Client. "
                "2. Implement a new InstanceFactory."
            )

    @staticmethod
    def restore_instance_for_http_transfer(
        instance_factory_type: InstanceFactoryType,
        parameter_dict: dict[str, Any],
    ) -> Any:
        match instance_factory_type:
            case InstanceFactoryType.SimpleImmutableType:
                module = InstanceFactoryUtility._get_type_str(
                    SimpleImmutableTypeInstanceFactory
                )
            case InstanceFactoryType.Enum:
                module = InstanceFactoryUtility._get_type_str(EnumInstanceFactory)
            case InstanceFactoryType.General:
                module = InstanceFactoryUtility._get_type_str(GeneralInstanceFactory)
            case _:
                raise NotImplementedError()
        instance_factory: InstanceFactoryInterface = GeneralInstanceFactory(
            module=module, parameters=parameter_dict
        ).create()  # Use GeneralInstanceFactory to create an InstanceFactory
        # Use the created InstanceFactory to create an instance
        return instance_factory.create()
