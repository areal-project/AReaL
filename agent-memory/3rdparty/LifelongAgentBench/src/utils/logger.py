import logging
import logging.config
import time
import datetime
import os
from typing import Optional, Any, Mapping, Callable

from src.typings import LoggerConfig
from .color_message import ColorMessage


class LoggerUtility:
    # Define emojis for different log levels
    _LEVEL_EMOJIS = {
        logging.DEBUG: "🐛",
        logging.INFO: "📘",
        logging.WARNING: "🚧",
        logging.ERROR: "🚨",
        logging.CRITICAL: "🔥",
    }
    # Mapping of log levels to their corresponding color methods
    _LEVEL_COLORS: Mapping[int, Callable[[str], str]] = {
        logging.DEBUG: ColorMessage.green,
        logging.INFO: ColorMessage.cyan,
        logging.WARNING: ColorMessage.yellow,
        logging.ERROR: ColorMessage.red,
        logging.CRITICAL: ColorMessage.magenta,
    }

    @staticmethod
    def construct_prefix(
        record: logging.LogRecord, start_time: float, colored_flag: bool
    ) -> str:
        elapsed_seconds = round(record.created - start_time)
        timestamp = datetime.datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        emoji = LoggerUtility._LEVEL_EMOJIS.get(record.levelno, "")
        level_name = record.levelname
        if colored_flag:
            colored_level = LoggerUtility.dye_string(level_name, record.levelno)
            return (
                f"{emoji} {ColorMessage.bold(colored_level):<25} | "
                f"{ColorMessage.green(timestamp)} | "
                f"{ColorMessage.blue(str(datetime.timedelta(seconds=elapsed_seconds)))}"
            )
        else:
            return (
                f"{emoji} {level_name:<8} | "
                f"{timestamp} | "
                f"{str(datetime.timedelta(seconds=elapsed_seconds))}"
            )

    @staticmethod
    def safely_get_message(record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:  # noqa
            message = record.msg
            if record.args:
                message += f" {record.args}"
        return message

    @staticmethod
    def load_logging_config(
        log_file_path: str, logging_level: str, logger_name: str
    ) -> dict[str, Any]:
        """
        Constructs and returns a logging configuration dictionary.

        Args:
            log_file_path (str): Path to the log file.
            logging_level (str): Logging level (e.g., 'INFO', 'DEBUG').
            logger_name (str): Name of the main application logger.

        Returns:
            dict: Logging configuration dictionary.
        """
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "colored": {
                    "()": ColoredLogFormatter,
                },
                "plain": {
                    "()": PlainLogFormatter,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "colored",
                    "level": logging_level,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "formatter": "plain",
                    "level": logging_level,
                    "maxBytes": 10 * 1024 * 1024,  # 10 MB
                    "backupCount": 50,  # 500 MB of logs is acceptable
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                logger_name: {  # Dynamically set the logger name
                    "handlers": ["console", "file"],
                    "level": logging_level,
                    "propagate": False,
                },
            },
        }

    @staticmethod
    def dye_string(string: str, logging_level: int) -> str:
        color_func = LoggerUtility._LEVEL_COLORS.get(logging_level, lambda x: x)
        return color_func(string)

    @staticmethod
    def beautify_multi_line_message(message: str, plain_prefix: str) -> str:
        logging_level_emoji_set = set(LoggerUtility._LEVEL_EMOJIS.values())
        additional_plain_prefix_length: int = 3  # Length of " | " after the prefix
        if plain_prefix[0] in logging_level_emoji_set:
            # The emoji will take the space of two characters, but len(emoji) == 1
            additional_plain_prefix_length += 1
        return message.replace(
            "\n", "\n" + " " * (len(plain_prefix) + additional_plain_prefix_length)
        )


class ColoredLogFormatter(logging.Formatter):
    """Custom formatter to include log level, timestamp, elapsed time with ANSI colors and emojis."""

    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.time()

    def format(self, record: logging.LogRecord) -> str:
        # region Prepare message
        message = LoggerUtility.safely_get_message(record)
        # endregion
        # region Apply color based on log level
        colored_message = LoggerUtility.dye_string(message, record.levelno)
        # endregion
        # region Handle multiline messages for better readability
        plain_prefix = LoggerUtility.construct_prefix(
            record, self.start_time, colored_flag=False
        )
        colored_message = LoggerUtility.beautify_multi_line_message(
            colored_message, plain_prefix
        )
        # endregion
        prefix = LoggerUtility.construct_prefix(
            record, self.start_time, colored_flag=True
        )
        return f"{prefix} | {colored_message}"


class PlainLogFormatter(logging.Formatter):
    """Plain formatter without ANSI colors."""

    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.time()

    def format(self, record: logging.LogRecord) -> str:
        prefix = LoggerUtility.construct_prefix(
            record, self.start_time, colored_flag=False
        )
        message = LoggerUtility.safely_get_message(record)
        # Handle multiline messages for better readability
        message = LoggerUtility.beautify_multi_line_message(message, prefix)
        return f"{prefix} | {message}"


class SingletonLogger:
    """
    Singleton class to manage a unified logger for the application.
    Ensures that only one instance of the logger exists throughout the application.
    """

    _instance: Optional["SingletonLogger"] = None

    def __init__(self, logger_name: str = "lifelong_agent_bench"):
        """
        Initializes the logger instance.

        Args:
            logger_name (str): Name of the logger.
        """
        self.logger = logging.getLogger(logger_name)

    @classmethod
    def get_instance(cls, config: Optional[LoggerConfig] = None) -> "SingletonLogger":
        """
        Retrieves the singleton instance of the logger. If it doesn't exist, initializes it.

        Args:
            config (LoggerConfig): Configuration object for initializing the logger.

        Returns:
            SingletonLogger: The singleton logger instance.

        Raises:
            ValueError: If no configuration is provided when creating the first instance.
        """
        if cls._instance is None:
            if not config:
                raise ValueError(
                    "LoggerConfig must be provided when creating the first instance."
                )
            # Ensure the log directory exists
            log_dir = os.path.dirname(config.log_file_path)
            os.makedirs(log_dir, exist_ok=True)
            # Load and apply the logging configuration with dynamic logger name
            logging_config = LoggerUtility.load_logging_config(
                config.log_file_path, config.level, config.logger_name
            )
            logging.config.dictConfig(logging_config)
            cls._instance = cls(config.logger_name)
        return cls._instance

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs an informational message."""
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a critical message."""
        self.logger.critical(msg, *args, **kwargs)


class SafeLogger:
    @staticmethod
    def _log_message(
        message: str, logging_level: int, *args: Any, **kwargs: Any
    ) -> None:
        try:
            logger = SingletonLogger.get_instance()
            match logging_level:
                case logging.DEBUG:
                    logger.debug(message, *args, **kwargs)
                case logging.INFO:
                    logger.info(message, *args, **kwargs)
                case logging.WARNING:
                    logger.warning(message, *args, **kwargs)
                case logging.ERROR:
                    logger.error(message, *args, **kwargs)
                case logging.CRITICAL:
                    logger.critical(message, *args, **kwargs)
                case _:
                    # Invalid logging level. Set to INFO.
                    logger.info(message, *args, **kwargs)
        except Exception as _:  # noqa
            # Do not print the redundant emoji
            print(LoggerUtility.dye_string(message, logging_level))

    @staticmethod
    def debug(msg: str, *args: Any, **kwargs: Any) -> None:
        SafeLogger._log_message(msg, logging.DEBUG, *args, **kwargs)

    @staticmethod
    def info(msg: str, *args: Any, **kwargs: Any) -> None:
        SafeLogger._log_message(msg, logging.INFO, *args, **kwargs)

    @staticmethod
    def warning(msg: str, *args: Any, **kwargs: Any) -> None:
        SafeLogger._log_message(msg, logging.WARNING, *args, **kwargs)

    @staticmethod
    def error(msg: str, *args: Any, **kwargs: Any) -> None:
        SafeLogger._log_message(msg, logging.ERROR, *args, **kwargs)

    @staticmethod
    def critical(msg: str, *args: Any, **kwargs: Any) -> None:
        SafeLogger._log_message(msg, logging.CRITICAL, *args, **kwargs)
