# SPDX-License-Identifier: Apache-2.0


class ModelExecutionTimeoutError(RuntimeError):
    """A backend confirmed that an already running generation exceeded its limit.

    Queue, transport, gateway and episode deadlines must never use this type.
    """

    code = "model_execution_timeout"
    message = "Model generation exceeded the inference backend running timeout"

    def __init__(self, message: str = message) -> None:
        if message != self.message:
            raise ValueError(
                "Model timeout errors require the canonical backend message"
            )
        super().__init__(self.message)


class EngineError(Exception):
    def __init__(self, error_type, sub_error, reason):
        self.error_type = error_type
        self.sub_error = sub_error
        self.reason = reason
        super().__init__(
            f"error_type: {error_type}, sub_error: {sub_error}, reason: {reason}"
        )


class FrameworkError(Exception):
    def __init__(self, error_type, sub_error, reason):
        self.error_type = error_type
        self.sub_error = sub_error
        self.reason = reason
        super().__init__(
            f"error_type: {error_type}, sub_error: {sub_error}, reason: {reason}"
        )
