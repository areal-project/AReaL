# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for the scheduler module."""


class SchedulerError(Exception):
    """Base exception for all scheduler-related errors."""


class WorkerCreationError(SchedulerError):
    """Raised when worker creation fails during subprocess spawn or initialization."""

    def __init__(self, worker_key: str, reason: str, details: str = ""):
        self.worker_key = worker_key
        self.reason = reason
        self.details = details
        message = f"Failed to create worker '{worker_key}': {reason}"
        if details:
            message += f"\nDetails: {details}"
        super().__init__(message)


class WorkerConfigurationError(SchedulerError):
    def __init__(self, worker_key: str, reason: str, details: str = ""):
        self.worker_key = worker_key
        self.reason = reason
        self.details = details
        message = f"Failed to configure worker '{worker_key}': {reason}"
        if details:
            message += f"\nDetails: {details}"
        super().__init__(message)


class WorkerFailedError(SchedulerError):
    """Raised when a worker process fails or exits unexpectedly."""

    def __init__(self, worker_id: str, exit_code: int, stderr: str = ""):
        self.worker_id = worker_id
        self.exit_code = exit_code
        self.stderr = stderr
        message = f"Worker '{worker_id}' failed with exit code {exit_code}"
        if stderr:
            message += f"\nStderr output:\n{stderr}"
        super().__init__(message)


class WorkerNotFoundError(SchedulerError):
    """Raised when attempting to access a worker that doesn't exist."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        super().__init__(f"Worker '{worker_id}' not found")


class EngineCreationError(SchedulerError):
    """Raised when engine creation fails on a worker."""

    def __init__(self, worker_id: str, reason: str, status_code: int = None):
        self.worker_id = worker_id
        self.reason = reason
        self.status_code = status_code
        message = f"Failed to create engine on worker '{worker_id}': {reason}"
        if status_code:
            message += f" (HTTP {status_code})"
        super().__init__(message)


class EngineCallError(SchedulerError):
    """Raised when calling an engine method fails."""

    def __init__(self, worker_id: str, method: str, reason: str, attempt: int = 1):
        self.worker_id = worker_id
        self.method = method
        self.reason = reason
        self.attempt = attempt
        message = f"Failed to call method '{method}' on worker '{worker_id}': {reason}"
        if attempt > 1:
            message += f" (after {attempt} attempts)"
        super().__init__(message)


class WorkerTimeoutError(SchedulerError):
    """Raised when waiting for a worker exceeds the timeout."""

    def __init__(self, worker_key: str, timeout: float):
        self.worker_key = worker_key
        self.timeout = timeout
        super().__init__(
            f"Timeout waiting for worker '{worker_key}' (waited {timeout}s)"
        )


class PortAllocationError(SchedulerError):
    """Raised when port allocation fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Failed to allocate ports: {reason}")


class GPUAllocationError(SchedulerError):
    """Raised when GPU allocation fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Failed to allocate GPU resources: {reason}")


class RPCConnectionError(SchedulerError):
    """Raised when RPC connection to a worker fails."""

    def __init__(self, worker_id: str, host: str, port: int, reason: str):
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.reason = reason
        super().__init__(
            f"Failed to connect to worker '{worker_id}' at {host}:{port}: {reason}"
        )


class EngineImportError(SchedulerError):
    """Raised when importing an engine class fails on the worker."""

    def __init__(self, import_path: str, reason: str):
        self.import_path = import_path
        self.reason = reason
        super().__init__(f"Failed to import engine '{import_path}': {reason}")
