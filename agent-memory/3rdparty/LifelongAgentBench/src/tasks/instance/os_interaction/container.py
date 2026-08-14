from docker.models import containers
import json
import threading
import docker

from .utility import (
    CommandItem,
    CommandName,
    CommandExecutionResult,
)


class OSInteractionContainer:
    def __init__(self, command_execution_timeout: int, image: str = "local-os/default"):
        client = docker.from_env()
        self.container: containers.Container = client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
            labels={"created_by": "os-pipeline"},
        )
        self.timeout_sec: float = command_execution_timeout

    def terminate(self) -> None:
        self.container.kill()

    def execute_independent(
        self, command_item: CommandItem, *parameters: str
    ) -> CommandExecutionResult:
        command_name = command_item.command_name
        script = command_item.script
        del command_item
        match command_name:
            case CommandName.BASH:
                cmd = ["bash", "-c", script]
                if parameters:
                    cmd.append("--")
                    cmd.extend(parameters)
            case CommandName.PYTHON:
                cmd = ["python3", "-c", script, *parameters]
            case CommandName.CPP:
                command_item = CommandItem(
                    command_name=CommandName.BASH,
                    script=f'echo "{json.dumps(script)}" > /tmp/main.cpp && '
                    f"g++ -o /tmp/a.out /tmp/main.cpp",
                )
                self.execute_independent(command_item)
                cmd = ["/tmp/a.out", *parameters]
            case CommandName.C:
                command_item = CommandItem(
                    command_name=CommandName.BASH,
                    script=f'echo "{json.dumps(script)}" > /tmp/main.cpp && '
                    f"gcc -o /tmp/a.out /tmp/main.cpp",
                )
                self.execute_independent(command_item)
                cmd = ["/tmp/a.out", *parameters]
            case _:
                raise NotImplementedError("Unsupported language")
        # Instead of directly calling `self.container.exec_run(cmd)`,
        # we wrap it with our thread-based timeout function.
        return self._execute_with_timeout(cmd)

    def _execute_with_timeout(
        self,
        cmd: list[str],
    ) -> CommandExecutionResult:
        """
        Runs `container.exec_run(cmd)` in a separate thread, enforcing a timeout.
        Raises TimeoutException if `timeout_sec` is exceeded.
        """
        result_holder: dict[str, CommandExecutionResult | Exception] = {}

        def run_exec() -> None:
            try:
                exec_result: docker.models.containers.ExecResult = (
                    self.container.exec_run(cmd)
                )
                result_holder["result"] = CommandExecutionResult(
                    exit_code=exec_result.exit_code,
                    output=exec_result.output.decode("utf-8"),
                    timeout_flag=False,
                )
            except Exception as e:
                result_holder["exception"] = e

        thread = threading.Thread(target=run_exec, daemon=True)
        thread.start()
        thread.join(self.timeout_sec)

        if thread.is_alive():
            # Thread didn't finish in time
            # It is not suitable to raise an exception here, since the timeout will not stop the interaction loop
            # between the agent and the task. This means that timeout is an excepted behavior and should not be handled
            # as an exception.
            return CommandExecutionResult(
                exit_code=None, output=None, timeout_flag=True
            )

        # If an exception was captured in the worker thread, re-raise it here
        if (exception := result_holder.get("exception")) is not None:
            assert isinstance(exception, Exception), "Check the code of run_exec()"
            raise exception
        # I do not think it is possible to throw KeyError here.
        # If it does happen, The caller should throw a TaskEnvironmentException.
        result = result_holder["result"]
        assert isinstance(result, CommandExecutionResult)
        return result
