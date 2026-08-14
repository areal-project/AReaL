from typing import Optional
import datetime
import traceback


class ContinualAgentBenchException(Exception):
    _record_file = None

    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__()
        self.detail = detail
        if self._record_file is None:
            return
        with open(self._record_file, "a") as f:
            f.write(f'Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f"Exception: {self.get_complete_description()}\n")
            f.write(f"{traceback.format_exc()}\n")
            f.write("\n")

    def get_complete_description(self) -> str:
        if self.detail is None:
            return f"[{self.__class__.__name__}]"
        else:
            return f"[{self.__class__.__name__}] {self.detail}"

    def __str__(self) -> str:
        maximum_detail_length = 4096  # Change it carefully
        if self.detail is None or len(self.detail) < maximum_detail_length:
            return self.get_complete_description()
        brief_detail_head = self.detail[: maximum_detail_length // 2]
        brief_detail_tail = self.detail[-maximum_detail_length // 2 :]
        omitted_detail_length = len(self.detail) - maximum_detail_length
        return (
            f"[{self.__class__.__name__}] "
            f"{brief_detail_head}"
            f"...({omitted_detail_length} characters omitted, "
            f"use e.get_complete_description() to see the full detail)..."
            f"{brief_detail_tail}"
        )

    @classmethod
    def from_exception(cls, e: Exception) -> "ContinualAgentBenchException":
        """
        Calling this function will construct a new exception.
        The exception will not be raised unless explicitly done so.
        Used to handle the exception that is not expected during coding.
        Such as AgentUnknownException, TaskEnvironmentException, TaskUnknownException
        """
        raise e  # Disable error handling
        return cls(f"{e.__class__.__name__}: {str(e)}")

    @classmethod
    def set_record_file(cls, record_file: str) -> None:
        cls._record_file = record_file


class ModelException(ContinualAgentBenchException):
    pass


class AgentException(ContinualAgentBenchException):
    pass


class TaskException(ContinualAgentBenchException):
    pass


class HttpException(ContinualAgentBenchException):
    pass


# region ModelException
class LanguageModelUnknownException(ModelException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


class LanguageModelContextLimitException(ModelException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


class LanguageModelOutOfMemoryException(ModelException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


# endregion
# region AgentException
class AgentUnknownException(AgentException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


class AgentContextLimitException(AgentException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


class AgentOutOfMemoryException(AgentException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


# endregion
# region TaskException
class TaskEnvironmentException(TaskException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


# Check whether it is used frequently. If not, replace it with TaskEnvironmentException
# 20241231: Used in DBBench._release().
class TaskReleaseException(TaskException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


class TaskUnknownException(TaskException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


# endregion
# region HttpException
class HttpTimeoutException(HttpException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


class HttpServerException(HttpException):
    def __init__(self, detail: Optional[str] = None) -> None:
        """
        500 <= status_code < 600
        The exception may be caused due to the error in the server or the forwarding app.
        """
        super().__init__(detail)


class HttpClientException(HttpException):
    def __init__(self, detail: Optional[str] = None) -> None:
        """
        400 <= status_code < 500
        The expected reason for this exception is the incorrect parameters in the request, which is almost impossible
        to happen.
        """
        super().__init__(detail)


class HttpUnknownException(HttpException):
    def __init__(self, detail: Optional[str] = None) -> None:
        super().__init__(detail)


# endregion
