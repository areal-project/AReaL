# SPDX-License-Identifier: Apache-2.0


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
