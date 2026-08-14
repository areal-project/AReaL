import argparse
import time
import requests
import subprocess

from src.utils import SingletonLogger, ConfigLoader
from src.run_experiment import ConfigUtility, ConfigUtilityCaller
from src.distributed_deployment_utils.server_side_controller import (
    StartServerRequest,
    StartServerResponse,
    ShutdownServerRequest,
    ShutdownServerResponse,
)
from src.typings import EnvironmentConfig


class ClientSideController:
    def __init__(
        self,
        logger: SingletonLogger,
        environment_config: EnvironmentConfig,
        config_path: str,
    ):
        self.logger = logger
        self.environment_config = environment_config
        self.config_path = config_path

    def _send_request_to_server_side_controller(
        self, data: StartServerRequest | ShutdownServerRequest
    ) -> bool:
        response_cls: type[StartServerResponse | ShutdownServerResponse]
        assert self.environment_config.server_side_controller_address is not None
        if isinstance(data, StartServerRequest):
            url = f"{self.environment_config.server_side_controller_address}/start_server/"
            response_cls = StartServerResponse
        elif isinstance(data, ShutdownServerRequest):
            url = f"{self.environment_config.server_side_controller_address}/shutdown_server/"
            response_cls = ShutdownServerResponse
        else:
            raise NotImplementedError()
        response = requests.post(
            url,
            json=data.model_dump(),
        )
        if response.ok and response_cls.model_validate(response.json()).success_flag:
            return True
        response_detail: str = ""
        try:
            # Format the response detail
            # Validate the response format
            response_detail = response_cls.model_validate(
                response.json()
            ).model_dump_json()
        except:  # noqa
            pass
        self.logger.error(
            f"Failed to send request to server side controller.\n"
            f"HTTP Status: {response.status_code}.\n"
            f"Response: {response.text}.\n"
            f"Response detail: {response_detail}."
        )
        return False

    def start_server(self) -> bool:
        self.logger.info(f"Sending the StartServerRequest to ServerSideController.")
        request = StartServerRequest(config_path=self.config_path)
        success_flag = self._send_request_to_server_side_controller(request)
        if success_flag:
            self.logger.info(f"Server started successfully.")
        else:
            self.logger.error(f"Failed to start server.")
        return success_flag

    def run_experiment(self) -> bool:
        self.logger.info("Running experiment.")
        assert self.environment_config.interpreter_path is not None
        result = subprocess.run(
            [
                self.environment_config.interpreter_path,
                "./src/run_experiment.py",
                "--config_path",
                self.config_path,
            ]
        )
        if result.returncode != 0:
            self.logger.error(f"Experiment with config {self.config_path} failed.")
            return False
        else:
            self.logger.info(f"Experiment with config {self.config_path} completed.")
            return True

    def shutdown_server(self) -> bool:
        self.logger.info(f"Sending the ShutdownServerRequest to ServerSideController.")
        request = ShutdownServerRequest()
        success_flag = self._send_request_to_server_side_controller(request)
        if success_flag:
            self.logger.info(f"Server shut down successfully.")
        else:
            self.logger.error(f"Failed to shut down server.")
        return success_flag


def main() -> None:
    # region Preparation
    # region Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to config from the project root.",
    )
    config_path = parser.parse_args().config_path
    # endregion
    raw_config = ConfigLoader().load_from(config_path)
    assert raw_config["environment_config"]["use_task_client_flag"]
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(
            raw_config, ConfigUtilityCaller.CLIENT_SIDE_CONTROLLER
        )
    )
    logger = SingletonLogger.get_instance(logger_config)
    client_side_controller = ClientSideController(
        logger, environment_config, config_path
    )
    sleep_duration: int = 15
    # endregion
    # region Start server
    logger.info(f"Starting experiment with config {config_path}.")
    start_server_success_flag = client_side_controller.start_server()
    if not start_server_success_flag:
        logger.error(f"Failed to start server.\n" f"config_path: {config_path}")
        return
    time.sleep(sleep_duration)
    # endregion
    # region Run experiment, shutdown server
    experiment_success_flag = client_side_controller.run_experiment()
    shutdown_server_success_flag = client_side_controller.shutdown_server()
    time.sleep(sleep_duration)
    # endregion
    logger.info(
        f"Experiment completed.\n"
        f"config_path: {config_path}\n"
        f"start_server_success_flag: {start_server_success_flag}\n"
        f"experiment_success_flag: {experiment_success_flag}\n"
        f"shutdown_server_success_flag: {shutdown_server_success_flag}"
    )


if __name__ == "__main__":
    main()
