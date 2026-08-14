from fastapi import APIRouter
from typing import Any

from src.typings import (
    GeneralRequest,
    GeneralResponse,
    InstanceFactoryUtility,
    InstanceFactoryType,
)
from .client import Client
from abc import ABC, abstractmethod


class Server(ABC):
    def __init__(self, router: APIRouter, principal: object):
        self.router = router
        self.principal = principal
        self.router.post("/ping")(self.ping)
        self.router.post("/get_attribute")(self.get_attribute)
        self.router.post("/set_attribute")(self.set_attribute)

    @staticmethod
    def ping() -> GeneralResponse.Ping:
        return GeneralResponse.Ping(response="Hello, World!")

    def get_attribute(
        self, data: GeneralRequest.GetAttribute
    ) -> GeneralResponse.GetAttribute:
        try:
            value = getattr(self.principal, data.name)
        except AttributeError:
            return GeneralResponse.GetAttribute(
                instance_factory_type=None,
                instance_factory_parameter_dict=None,
            )
        instance_factory, instance_factory_type = (
            InstanceFactoryUtility.create_instance_factory_for_http_transfer(value)
        )
        # TODO: Verify whether it is necessary
        #   Currently, the if statement is kept to ensure the logic is correct.
        if instance_factory_type == InstanceFactoryType.General:
            assert isinstance(value, Client)
        return GeneralResponse.GetAttribute(
            instance_factory_type=instance_factory_type,
            instance_factory_parameter_dict=instance_factory.model_dump(),
        )

    def set_attribute(self, data: GeneralRequest.SetAttribute) -> None:
        value = InstanceFactoryUtility.restore_instance_for_http_transfer(
            instance_factory_type=data.instance_factory_type,
            parameter_dict=data.instance_factory_parameter_dict,
        )
        setattr(self.principal, data.name, value)
        return

    @staticmethod
    @abstractmethod
    def start_server(*args: Any, **kwargs: Any) -> None:
        pass
