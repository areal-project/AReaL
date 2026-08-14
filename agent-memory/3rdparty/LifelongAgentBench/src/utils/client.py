from typing import Any, Optional, Type, TypeVar, overload, reveal_type
import requests
from pydantic import BaseModel

from src.typings import (
    GeneralRequest,
    GeneralResponse,
    InstanceFactoryUtility,
    HttpException,
    HttpTimeoutException,
    HttpServerException,
    HttpClientException,
    HttpUnknownException,
)
from .retry import RetryHandler, ExponentialBackoffStrategy
from .logger import SafeLogger


T = TypeVar("T", bound=BaseModel)


class Client(BaseModel):
    server_address: str
    request_timeout: int

    def __init__(self, **data: Any):
        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization processing to clean server address and verify connectivity.
        Removes trailing slash from the server address and pings the server to ensure connectivity.
        """
        if self.server_address.endswith("/"):
            self.server_address = self.server_address.rstrip("/")
        # Verify connectivity
        try:
            _ = self._call_server("/ping", None, GeneralResponse.Ping)
        except HttpException as e:
            error_message = (
                f"Cannot initialize {type(self).__name__}.\n"
                f"Reason: Failed to ping server at {self.server_address}.\n"
            )
            SafeLogger.error(error_message)
            raise e.__class__(error_message) from e
        except Exception as e:
            error_message = (
                f"Cannot initialize {type(self).__name__}.\n"
                f"Reason: Unknown error occurs when pinging server at {self.server_address}.\n"
            )
            SafeLogger.error(error_message)
            raise HttpUnknownException(error_message) from e

    @overload
    def _call_server(
        self,
        api: str,
        data: Optional[BaseModel],
        response_cls: Type[T],
    ) -> T:
        pass

    @overload
    def _call_server(
        self,
        api: str,
        data: Optional[BaseModel],
        response_cls: None,
    ) -> None:
        pass

    @RetryHandler.handle(
        max_retries=20,
        retry_on=(
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            HttpException,
        ),
        waiting_strategy=ExponentialBackoffStrategy(interval=(None, 30), multiplier=2),
    )  # The retry will take at most 10 minutes.
    def _call_server(
        self,
        api: str,
        data: Optional[BaseModel],
        response_cls: Optional[Type[T]],
    ) -> Optional[T]:
        """
        Args:
            api (str): The API endpoint (must start with '/').
            data (Optional[BaseModel]): The request payload.
            response_cls (Optional[Type[BaseModel]]): The class to validate the response against.

        Returns:
            Optional[BaseModel]: The validated response object or None.
        """
        # region Preparation
        assert api.startswith("/")
        address = self.server_address + api
        if data is None:
            data_dict = {}
        else:
            data_dict = data.model_dump()
        request_info_str = (
            f"Request information:\n"
            f"- address: {address}\n"
            f"- response_cls: {response_cls}\n"
            f"- data_cls: {str(type(data))}\n"
            f"- data_dict_str_length: {len(str(data_dict))}\n"
            f"- data_dict: {data_dict}"
        )
        error_description_placeholder = "<<ERROR_DESCRIPTION>>"
        error_message_template = f"{error_description_placeholder}\n{request_info_str}"
        # endregion
        # region Send request
        try:
            response = requests.post(
                address, json=data_dict, timeout=self.request_timeout
            )
        except requests.exceptions.Timeout as e:
            error_message = error_message_template.replace(
                error_description_placeholder, "Request timeout."
            )
            SafeLogger.error(error_message)
            raise HttpTimeoutException(error_message) from e
        except requests.exceptions.ConnectionError as e:
            error_message = error_message_template.replace(
                error_description_placeholder,
                "Error occurs when reaching the destination server. Do you start the server?",
            )
            SafeLogger.error(error_message)
            raise HttpServerException(error_message) from e
        except Exception as e:
            error_message = error_message_template.replace(
                error_description_placeholder,
                "Unknown error occurs when sending request.",
            )
            SafeLogger.error(error_message)
            raise HttpUnknownException(error_message) from e
        # endregion
        # region Process response
        if response.ok:  # response will always be assigned in the try block.
            if response_cls is None:
                return None
            response_dict = response.json()
            return response_cls.model_validate(response_dict)
        else:
            error_info_str = (
                "Error information:\n"
                f"- status_code: {response.status_code}\n"
                f"- text: {response.text}"
            )
            try:
                response.raise_for_status()  # The statement will definitely raise requests.exceptions.HTTPError.
            except requests.exceptions.HTTPError as e:
                if 400 <= response.status_code < 600:
                    error_message = error_message_template.replace(
                        error_description_placeholder,
                        f"Original error message: {e}\n{error_info_str}",
                    )
                    SafeLogger.error(error_message)
                    if 400 <= response.status_code < 500:
                        raise HttpClientException(error_message) from e
                    else:
                        raise HttpServerException(error_message) from e
                else:
                    # This block is not expected to be triggered. Since `response.raise_for_status()` will only raise
                    # requests.exceptions.HTTPError when the status code is in of the range of [400, 600).
                    error_message = error_message_template.replace(
                        error_description_placeholder,
                        f"The status code of the request is out of the range of [400, 600).\n{error_info_str}",
                    )
                    SafeLogger.error(error_message)
                    raise HttpUnknownException(error_message) from e
            except Exception as e:
                # This block is not expected to be triggered. Since ``response.raise_for_status()`` will only raise
                # requests.exceptions.HTTPError.
                error_message = error_message_template.replace(
                    error_description_placeholder,
                    f"`response.raise_for_status()` raises an unknown error.\n{error_info_str}",
                )
                SafeLogger.error(error_message)
                raise HttpUnknownException(error_message) from e
            return None  # Will never reach here, added for type checking
        # endregion

    def __getattr__(self, name: str) -> Any:
        """
        Args:
            name (str): The attribute name to retrieve.

        Returns:
            Any: The retrieved attribute.
        """
        # https://stackoverflow.com/questions/4295678/
        excluded_attributes = {"shape", "__len__"}
        if name in excluded_attributes:
            # It seems that a these two names is frequently called by other packages (or the interpreter?), but I
            # cannot find out the callers.
            # I cannot even stop the process when names are assigned by these two values.
            # So I have to add this to prevent the server from being called if the name is "shape" or "__len__".
            return super().__getattr__(name)  # type: ignore[misc]
        response: GeneralResponse.GetAttribute = self._call_server(
            "/get_attribute",
            GeneralRequest.GetAttribute(name=name),
            GeneralResponse.GetAttribute,
        )
        if response.instance_factory_type is None:
            SafeLogger.error(f"Attribute '{name}' not found on the server.")
            raise AttributeError(f"Attribute '{name}' is not found.")
        if response.instance_factory_parameter_dict is None:
            raise RuntimeError(
                "The instance factory has no parameter, which is unexpected."
            )
        return InstanceFactoryUtility.restore_instance_for_http_transfer(
            instance_factory_type=response.instance_factory_type,
            parameter_dict=response.instance_factory_parameter_dict,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Args:
            name (str): The attribute name to set.
            value (Any): The value to assign to the attribute.
        """
        instance_factory, instance_factory_type = (
            InstanceFactoryUtility.create_instance_factory_for_http_transfer(value)
        )
        _ = self._call_server(
            "/set_attribute",
            GeneralRequest.SetAttribute(
                name=name,
                instance_factory_type=instance_factory_type,
                instance_factory_parameter_dict=instance_factory.model_dump(),
            ),
            None,
        )

    def __delattr__(self, name: str) -> None:
        """
        Prevents deletion of attributes by raising a RuntimeError.

        Args:
            name (str): The attribute name to delete.
        """
        SafeLogger.error(f"Attempt to delete attribute '{name}' was blocked.")
        raise RuntimeError("Attribute deletion is not allowed.")
