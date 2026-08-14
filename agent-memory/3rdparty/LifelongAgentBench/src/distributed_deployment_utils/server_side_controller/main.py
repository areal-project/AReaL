from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import subprocess
import uvicorn
import traceback
import time
import docker
import psutil
from typing import Optional

from src.typings import LoggerConfig
from src.utils import SingletonLogger
from src.distributed_deployment_utils.server_side_controller.utility import (
    StartServerRequest,
    StartServerResponse,
    ShutdownServerRequest,
    ShutdownServerResponse,
)


class ServerSideController:
    def __init__(self, logger: SingletonLogger) -> None:
        self.logger = logger
        self.app = FastAPI()
        self.router = APIRouter()
        self.router.post("/start_server/")(self.start_server)
        self.router.post("/shutdown_server/")(self.shutdown_server)
        self.app.include_router(self.router)
        # State variables
        self.server_up_flag = False
        self.server_related_docker_container_id_list: list[str] = []
        self.server_pid: Optional[int] = None
        self.logger.info("ServerSideController initialized.")

    def start_server(self, request: StartServerRequest) -> StartServerResponse:
        # region Preparation
        self.logger.info(f"Received request to start server. Request: {request}")
        if self.server_up_flag:
            self.logger.error("Server already started. Returning.")
            return StartServerResponse(
                success_flag=False, message="Server already started"
            )
        command = [
            "python",
            "./src/distributed_deployment_utils/start_server.py",
            "--config_path",
            request.config_path,
        ]
        client = docker.from_env()
        container_id_list_before = [c.id for c in client.containers.list()]
        # endregion
        # region Execute start_server.py to start the server
        self.logger.info(f"Executing command (Popen): {' '.join(command)}")
        try:
            server_process = subprocess.Popen(command)
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            return StartServerResponse(success_flag=False, message=str(e))
        self.logger.info(f"Started server with PID: {server_process.pid}")
        # endregion
        # region Check the status of the server process
        time.sleep(15)  # Sleep for some time to give the server a chance to start
        server_running_flag: bool
        if psutil.pid_exists(server_process.pid):
            process = psutil.Process(server_process.pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                server_running_flag = False
            else:
                server_running_flag = True
        else:
            server_running_flag = False
        if not server_running_flag:
            error_message = (
                f"Server process with PID {server_process.pid} exited prematurely."
            )
            self.logger.error(error_message)
            return StartServerResponse(success_flag=False, message=error_message)
        # endregion
        # region Maintain state, return response
        self.logger.info("Server started successfully.")
        container_id_list_after = [c.id for c in client.containers.list()]
        self.server_related_docker_container_id_list = list(
            set(container_id_list_after) - set(container_id_list_before)
        )
        self.logger.info(f"{self.server_related_docker_container_id_list=}")
        self.server_up_flag = True
        self.server_pid = server_process.pid
        return StartServerResponse(
            success_flag=True, message="Server started successfully"
        )
        # endregion

    def shutdown_server(self, request: ShutdownServerRequest) -> ShutdownServerResponse:
        # region Check if the server is already down
        if not self.server_up_flag:
            self.logger.error("Server already stopped. Returning.")
            return ShutdownServerResponse(
                success_flag=False, message="Server already stopped"
            )
        self.logger.info(f"Received request to shutdown server.")
        # endregion
        # region Plan A: Attempt graceful shutdown by PID (including child processes)
        graceful_shutdown_success: bool
        if self.server_pid and psutil.pid_exists(self.server_pid):
            # region Kill the parent process and all its children
            self.logger.info(
                f"Terminating server with PID {self.server_pid} and its children"
            )
            try:
                ServerSideController._kill_process_and_children(self.server_pid)
                graceful_shutdown_success = True
            except Exception as e:
                self.logger.error(f"Error shutting down server using plan A: {e}")
                graceful_shutdown_success = False
            # endregion
        else:
            self.logger.error(
                f"Server process with PID {self.server_pid} does not exist."
            )
            graceful_shutdown_success = False
        # endregion
        # region Plan B: Fallback to the existing shutdown_server.py script if Plan A failed
        if not graceful_shutdown_success:
            self.logger.info(
                "Graceful PID shutdown failed; falling back to shutdown_server.py script."
            )
            command = (
                f"python ./src/distributed_deployment_utils/shutdown_server.py "
                f"--process_name start_server.py "
                f"--docker_container_id_list {'_'.join(self.server_related_docker_container_id_list)} "
                f"--auto_confirm"
            )
            self.logger.info(f"Executing command: {command}")
            try:
                subprocess.run(command, shell=True, check=True)
                self.logger.info("Fallback to shutdown_server.py successful.")
            except Exception as e:
                self._reset_state()
                error_message = (
                    f"Error occurred while shutting down the server using plan B: {e}, "
                    f"The state of ServerSideController has been reset."
                )
                self.logger.error(error_message)
                return ShutdownServerResponse(success_flag=False, message=error_message)
        # endregion
        # region Shutdown successful
        self._reset_state()
        self.logger.info("Server shutdown successful.")
        return ShutdownServerResponse(
            success_flag=True, message="Server shutdown successful"
        )
        # endregion

    @staticmethod
    def _kill_process_and_children(pid: int) -> None:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        gone, alive = psutil.wait_procs(children + [parent], timeout=5.0)
        for p in alive:
            p.kill()

    def _reset_state(self) -> None:
        self.server_up_flag = False
        self.server_pid = None
        self.server_related_docker_container_id_list = []


def main() -> None:
    logger_config = LoggerConfig(
        level="DEBUG",
        log_file_path="./outputs/server_sider_controller.log",
        logger_name="server_side_controller",
    )
    logger = SingletonLogger.get_instance(logger_config)
    logger.info("Starting ServerSideController ...")
    try:
        server_side_controller = ServerSideController(logger)
        uvicorn.run(
            server_side_controller.app, host="0.0.0.0", port=8003, log_level="info"
        )
    except Exception:  # noqa
        logger.error(
            f"Error occurred while starting FastAPI app: {traceback.format_exc()}"
        )
        exit(1)


if __name__ == "__main__":
    main()
