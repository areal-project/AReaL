import argparse
import re
from multiprocessing import Process
import time

from src.tasks import Task, TaskServer, DatasetItem
from src.factories import ChatHistoryItemFactoryServer
from src.typings import GeneralInstanceFactory
from src.utils import ConfigLoader, SingletonLogger
from src.run_experiment import ConfigUtility, ConfigUtilityCaller


class ServerStarterUtility:
    search_port_pattern = r"https?://[^:]+:(\d+)"
    search_prefix_pattern = r"https?://[^/]+(/.*)"

    @classmethod
    def extract_server_port(cls, server_address: str) -> int:
        port_match = re.match(cls.search_port_pattern, server_address)
        if port_match is None:
            raise ValueError(f"Port not found in server address: {server_address}")
        return int(port_match.group(1))

    @classmethod
    def extract_server_prefix(cls, server_address: str) -> str:
        prefix_match = re.match(cls.search_prefix_pattern, server_address)
        if prefix_match is None:
            raise ValueError(f"Prefix not found in server address: {server_address}")
        return prefix_match.group(1)


def main() -> None:
    # region Read config
    # region Read config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    raw_config = ConfigLoader().load_from(args.config_path)
    assert raw_config["environment_config"][
        "use_task_client_flag"
    ], "Task client is not enabled in configuration."
    # endregion
    # region Initialize logger
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.SERVER)
    )
    logger = SingletonLogger.get_instance(logger_config)
    # endregion
    # region Read task_server config
    assert environment_config.task_client is not None
    task_server_address = environment_config.task_client.parameters["server_address"]
    task_server_port = ServerStarterUtility.extract_server_port(task_server_address)
    task_server_prefix = ServerStarterUtility.extract_server_prefix(task_server_address)
    task_instance_factory = assignment_config.task
    # endregion
    # region Read chat_history_item_factory config
    assert environment_config.chat_history_item_factory_client is not None
    chat_history_item_factory_server_address = (
        environment_config.chat_history_item_factory_client.parameters["server_address"]
    )
    chat_history_item_factory_server_port = ServerStarterUtility.extract_server_port(
        chat_history_item_factory_server_address
    )
    chat_history_item_factory_server_prefix = (
        ServerStarterUtility.extract_server_prefix(
            chat_history_item_factory_server_address
        )
    )
    chat_history_item_factory_instance_factory = GeneralInstanceFactory.model_validate(
        task_instance_factory.parameters["chat_history_item_factory"]
    )
    # endregion
    # endregion

    # region Replace chat_history_item_factory_client with its corresponding client
    task_instance_factory.parameters["chat_history_item_factory"] = (
        environment_config.chat_history_item_factory_client
    )
    # endregion

    process_information_list: list[tuple[Process, str]] = []
    try:
        # region start server
        # region start chat_history_item_factory_server
        logger.info("Starting Chat History Item Factory Server...")
        chat_history_item_factory = chat_history_item_factory_instance_factory.create()
        chat_history_item_factory_server_process = Process(
            target=ChatHistoryItemFactoryServer.start_server,
            args=(
                chat_history_item_factory,
                chat_history_item_factory_server_port,
                chat_history_item_factory_server_prefix,
            ),
            daemon=True,  # Ensure subprocess terminates with the main process
        )
        chat_history_item_factory_server_process.start()
        logger.info(f"Chat History Item Factory Server started.")
        process_information_list.append(
            (
                chat_history_item_factory_server_process,
                "Chat History Item Factory Server",
            )
        )
        time.sleep(5)  # wait for the server to start
        # endregion
        # region start task_server
        logger.info("Starting Task Server...")
        task: Task[DatasetItem] = task_instance_factory.create()
        task_server_process = Process(
            target=TaskServer.start_server,
            args=(task, task_server_port, task_server_prefix),
            daemon=True,  # Ensure subprocess terminates with the main process
        )
        task_server_process.start()
        logger.info(f"Task Server started.")
        process_information_list.append((task_server_process, "Task Server"))
        # endregion
        # endregion

        # region Monitor and keep the main process alive
        logger.info(
            "Both Task Server and Chat History Item Factory Server are running. Press Ctrl+C to exit."
        )
        while True:
            time.sleep(1)  # Use sleep instead of input() for better signal handling
        # endregion
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down servers...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # region Terminate subprocesses if they are still running
        for process, server_name in process_information_list:
            if process.is_alive():
                logger.info(f"Terminating {server_name}...")
                process.terminate()
                process.join()
                logger.info(f"{server_name} terminated.")
        logger.info("All servers have been shut down.")
        # endregion


if __name__ == "__main__":
    main()
