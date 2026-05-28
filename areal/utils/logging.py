# SPDX-License-Identifier: Apache-2.0

import logging.config
import os
import threading
from logging import WARNING, FileHandler, Logger, Manager, RootLogger
from typing import Literal

import colorlog
import colorlog.escape_codes
import colorlog.formatter

# ANSI color codes for the (AReaL) header
# Using 256-color mode for a milk tea / brown-yellow color (RGB ~180, 140, 80)
AREAL_HEADER = "\033[1;38;2;180;140;80m(AReaL)\033[0m"  # Bold milk tea color
AREAL_HEADER_PLAIN = "(AReaL)"  # For file logging (no colors)

LOG_FORMAT = f"{AREAL_HEADER} %(log_color)s%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
LOG_FORMAT_PLAIN = (
    f"{AREAL_HEADER_PLAIN} %(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
)
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
LOGLEVEL = logging.INFO
LOG_PREFIX_WIDTH = 10  # Fixed width for alignment in merged.log

# NOTE: To use colorlog we should not call colorama.init() anywhere.
# The available color names are black, red, green, yellow, blue, purple, cyan and white

# Logger color mappings by component category
# Exact matches take priority, then prefix patterns are checked in order
#
# Color scheme:
#   - blue: Schedulers, Launchers (infrastructure)
#   - white: Controllers, RPC, Inference wrappers (orchestration)
#   - light_purple/purple: Workflows, Rewards, OpenAI (RL-specific)
#   - light_green: Stats, Perf, Dataset, Trainers (data/metrics)
#   - light_cyan/cyan: Engines, Platforms, MCore (compute backends)
LOGGER_COLORS_EXACT = {
    # Schedulers - blue
    "LocalScheduler": "blue",
    "RayScheduler": "blue",
    "SlurmScheduler": "blue",
    # Launchers - blue
    "LocalLauncher": "blue",
    "RayLauncher": "blue",
    "SlurmLauncher": "blue",
    # Workflows - purple
    "RLVRWorkflow": "light_purple",
    "VisionRLVRWorkflow": "light_purple",
    "MultiTurnWorkflow": "light_purple",
    "MultiTurnV2Workflow": "light_purple",
    # Controllers - white
    "TrainController": "white",
    "RolloutController": "white",
    "WorkflowExecutor": "white",
    # Stats/Perf - green
    "StatsLogger": "light_green",
    "StatsTracker": "light_green",
    "PerfTracer": "light_green",
    # RPC servers - white
    "SyncRPCServer": "white",
    "RayRPCServer": "white",
    "RPCSerialization": "white",
    "HttpRTensor": "white",
    # Inference wrappers - white
    "SGLangWrapper": "white",
    "VLLMWrapper": "white",
    "RemoteInfEngine": "white",
    "vLLMEngine": "white",
    # Dataset - green
    "Dataset": "light_green",
    "CLEVR70KDataset": "light_green",
    # Trainers - green
    "RLTrainer": "light_green",
    "SFTTrainer": "light_green",
    # Algorithm-specific - cyan
    "PPOActor": "cyan",
    # Rewards - purple
    "GSM8KReward": "purple",
    "Geometry3KReward": "purple",
    "RewardUtils": "purple",
    "RewardAPI": "purple",
    # Tree attention - cyan
    "TreeAttentionWrapper": "light_cyan",
    "TreeAttentionFSDP": "light_cyan",
    "TreeAttentionMegatron": "light_cyan",
    "TreeAttentionCore": "light_cyan",
    "TreeAttentionConstants": "light_cyan",
    "TreeAttentionViz": "light_cyan",
    # Checkpoint - blue (infrastructure)
    "Saver": "blue",
    "AsyncCheckpoint": "blue",
    "ArchonCheckpoint": "blue",
    # Platforms - cyan
    "Platform": "light_cyan",
    "PlatformInit": "light_cyan",
    "CUDAPlatform": "light_cyan",
    "NPUPlatform": "light_cyan",
    "UnknownPlatform": "light_cyan",
    # Sandbox backends
    "DaytonaClientManager": "blue",
    "DaytonaRunner": "light_cyan",
    "DaytonaPythonTool": "light_purple",
    # OpenAI - purple
    "OpenAIClient": "light_purple",
    "OpenAICache": "light_purple",
    "OpenAIProxy": "light_purple",
    "ToolCallParser": "light_purple",
    "TokenLogpReward": "light_purple",
    "ProxyUtils": "light_purple",
    # Agent Service - purple
    "AgentGateway": "light_purple",
    "AgentBridge": "light_purple",
    "AgentRouter": "light_purple",
    "AgentWorker": "light_purple",
    "AgentDataProxy": "light_purple",
    "AgentController": "light_purple",
    # Inference service - white (orchestration)
    "RolloutControllerV2": "white",
    "InferenceDataProxy": "white",
    "InferenceInfBridge": "white",
    "InferenceRouter": "white",
    "InferenceGateway": "white",
    "RPCGuard": "white",
}

# Prefix patterns checked in order (first match wins)
# Used for dynamic logger names like "[FSDPEngine Rank 0]"
LOGGER_PATTERNS = [
    # Engines - cyan
    ("FSDPEngine", "light_cyan"),
    ("MegatronEngine", "light_cyan"),
    ("RemoteInfEngine", "light_cyan"),
    ("MCore", "light_cyan"),
    # HF utilities - white
    ("HF", "white"),
    # Tests - white
    ("Test", "white"),
]

DEFAULT_LOGGER_COLOR = "white"

# Store file handlers that should persist across getLogger() calls
_file_handlers: list[FileHandler] = []
# Track all loggers created via getLogger() so we can add file handlers to them
_created_loggers: dict[str, Logger] = {}
# Lock for thread-safe access to _file_handlers and _created_loggers
_loggers_lock = threading.Lock()


class StreamingFileHandler(FileHandler):
    """FileHandler that flushes after each log message for real-time streaming."""

    def emit(self, record):
        super().emit(record)
        self.flush()


class LoggerColoredFormatter(colorlog.ColoredFormatter):
    """Custom formatter that colors logs based on logger name for INFO/DEBUG levels.

    WARNING, ERROR, and CRITICAL levels keep their standard colors (yellow, red)
    to ensure they always stand out regardless of the source component.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger_color_cache: dict[str, str] = {}

    def _get_logger_color(self, name: str) -> str:
        """Get the color for a logger name, using cache for performance."""
        if name in self._logger_color_cache:
            return self._logger_color_cache[name]

        # Check exact matches first
        if name in LOGGER_COLORS_EXACT:
            color = LOGGER_COLORS_EXACT[name]
        else:
            # Check prefix patterns
            color = DEFAULT_LOGGER_COLOR
            for pattern, pattern_color in LOGGER_PATTERNS:
                if name.startswith(pattern) or pattern in name:
                    color = pattern_color
                    break

        self._logger_color_cache[name] = color
        return color

    def formatMessage(self, record):
        """Thread-safe formatting that uses per-logger colors for DEBUG/INFO."""
        # Get escape codes from parent's method
        escapes = self._escape_code_map(record.levelname)

        # For DEBUG/INFO, override with logger-based color (thread-safe since
        # we modify a local dict copy, not shared self.log_colors)
        if record.levelno < logging.WARNING:
            logger_color = self._get_logger_color(record.name)
            escapes = dict(escapes)  # Make a mutable copy
            escapes["log_color"] = colorlog.escape_codes.parse_colors(logger_color)

        wrapper = colorlog.formatter.ColoredRecord(record, escapes)
        message = super(colorlog.ColoredFormatter, self).formatMessage(wrapper)
        message = self._append_reset(message, escapes)
        return message


log_config = {
    "version": 1,
    "formatters": {
        "plain": {
            "()": LoggerColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored": {
            "()": LoggerColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "blue",
                "INFO": "light_purple",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored_system": {
            "()": LoggerColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "blue",
                "INFO": "light_green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
        "colored_benchmark": {
            "()": LoggerColoredFormatter,
            "format": "%(log_color)s" + LOG_FORMAT,
            "datefmt": DATE_FORMAT,
            "log_colors": {
                "DEBUG": "light_black",
                "INFO": "light_cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        },
    },
    "handlers": {
        "plainHandler": {
            "class": "logging.StreamHandler",
            "level": LOGLEVEL,
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
        "benchmarkHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "colored_benchmark",
            "stream": "ext://sys.stdout",
        },
        "systemHandler": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "colored_system",
            "stream": "ext://sys.stdout",
        },
        "coloredHandler": {
            "class": "logging.StreamHandler",
            "level": LOGLEVEL,
            "formatter": "colored",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "plain": {
            "handlers": ["plainHandler"],
            "level": LOGLEVEL,
        },
        "benchmark": {
            "handlers": ["benchmarkHandler"],
            "level": "DEBUG",
        },
        "colored": {
            "handlers": ["coloredHandler"],
            "level": LOGLEVEL,
        },
        "system": {
            "handlers": ["systemHandler"],
            "level": LOGLEVEL,
        },
        # Suppress verbose HTTP loggers from third-party libraries.
        "uvicorn": {"handlers": [], "level": "WARNING", "propagate": False},
        "uvicorn.access": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
        "httpx": {"handlers": [], "level": "WARNING", "propagate": False},
        "httpcore": {"handlers": [], "level": "WARNING", "propagate": False},
        "aiohttp": {"handlers": [], "level": "WARNING", "propagate": False},
        "aiohttp.access": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
        "aiohttp.server": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
        "werkzeug": {"handlers": [], "level": "WARNING", "propagate": False},
        "urllib3": {"handlers": [], "level": "WARNING", "propagate": False},
        "urllib3.connectionpool": {
            "handlers": [],
            "level": "WARNING",
            "propagate": False,
        },
    },
    "disable_existing_loggers": True,
}


def getLogger(
    name: str | None = None,
    type_: Literal["plain", "benchmark", "colored", "system"] | None = None,
    level: int = LOGLEVEL,
):
    # Fix the logging config automatically set by transformer_engine
    # by reset config everytime getLogger is called.
    root = RootLogger(WARNING)
    Logger.root = root
    Logger.manager = Manager(Logger.root)

    logging.config.dictConfig(log_config)

    if name is None:
        name = "plain"
    if type_ is None:
        type_ = "plain"
    assert type_ in ["plain", "benchmark", "colored", "system"]
    if name not in log_config["loggers"]:
        log_config["loggers"][name] = {
            "handlers": [f"{type_}Handler"],
            "level": level,
        }
        logging.config.dictConfig(log_config)

    logger = logging.getLogger(name)

    # Track this logger and add file handlers if setup_file_logging() was called.
    # We add handlers directly to each logger (not just root) because getLogger()
    # resets Logger.root each time, orphaning previously created loggers from the
    # new root logger that has the file handlers.
    with _loggers_lock:
        _created_loggers[name] = logger
        for handler in _file_handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)

    return logger


_HTTP_LOGGERS = (
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "httpx",
    "httpcore",
    "aiohttp",
    "aiohttp.access",
    "aiohttp.server",
    "werkzeug",
    "urllib3",
    "urllib3.connectionpool",
)


def suppress_http_loggers() -> None:
    """Force all HTTP-related loggers to WARNING.

    Call this from service __main__.py right before uvicorn.run() or
    app.run() to ensure third-party HTTP loggers stay quiet even after
    uvicorn/Flask reconfigure the logging hierarchy.
    """
    import logging as _logging

    for name in _HTTP_LOGGERS:
        _logging.getLogger(name).setLevel(_logging.WARNING)


def setup_file_logging(
    log_dir: str,
    filename: str = "main.log",
    level: int = LOGLEVEL,
) -> None:
    """Set up file logging for the controller process.

    Adds FileHandlers to all loggers so they write to:
    1. A dedicated log file (e.g., main.log) with ANSI colors
    2. A merged log file (merged.log) with a source prefix and ANSI colors

    This function adds handlers to all loggers that were previously created via
    getLogger(), and stores the handlers so future loggers also get them.

    Args:
        log_dir: Directory to write log files.
        filename: Log file name (default: main.log).
        level: Logging level.
    """
    # Ensure idempotency: only set up file logging once
    if _file_handlers:
        return

    os.makedirs(log_dir, exist_ok=True)

    # Handler for dedicated log file (with ANSI colors, same as stdout)
    # Uses StreamingFileHandler to flush after each message for real-time output
    file_handler = StreamingFileHandler(os.path.join(log_dir, filename), mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        LoggerColoredFormatter(
            LOG_FORMAT,
            datefmt=DATE_FORMAT,
            log_colors={
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        )
    )
    _file_handlers.append(file_handler)

    # Handler for merged.log (with fixed-width [main] prefix and ANSI colors)
    prefix = "[main]".ljust(LOG_PREFIX_WIDTH)
    merged_format = prefix + LOG_FORMAT
    merged_handler = StreamingFileHandler(os.path.join(log_dir, "merged.log"), mode="a")
    merged_handler.setLevel(level)
    merged_handler.setFormatter(
        LoggerColoredFormatter(
            merged_format,
            datefmt=DATE_FORMAT,
            log_colors={
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_white,bg_red",
            },
        )
    )
    _file_handlers.append(merged_handler)

    # Add file handlers to all previously created loggers
    with _loggers_lock:
        for logger in _created_loggers.values():
            for handler in [file_handler, merged_handler]:
                if handler not in logger.handlers:
                    logger.addHandler(handler)


_LATEST_LOG_STEP = 0


def log_swanlab_wandb_tensorboard(data, step=None, summary_writer=None):
    # Logs data to SwanLab, wandb, TensorBoard, and Trackio.

    global _LATEST_LOG_STEP
    if step is None:
        step = _LATEST_LOG_STEP
    else:
        _LATEST_LOG_STEP = max(_LATEST_LOG_STEP, step)

    # swanlab
    try:
        import swanlab

        swanlab.log(data, step=step)
    except (ModuleNotFoundError, ImportError):
        pass

    # wandb
    import wandb

    wandb.log(data, step=step)

    # trackio
    try:
        import trackio

        trackio.log(data, step=step)
    except (ModuleNotFoundError, ImportError):
        pass

    # tensorboard
    if summary_writer is not None:
        for key, val in data.items():
            summary_writer.add_scalar(f"{key}", val, step)


if __name__ == "__main__":
    # Test per-logger color differentiation
    # Run with: python -m areal.utils.logging
    print("=" * 70)
    print("Testing per-logger color differentiation with (AReaL) prefix")
    print("Each component category should have a distinct color:")
    print("  - blue: Schedulers, Launchers (infrastructure)")
    print("  - white: Controllers, RPC, Inference (orchestration)")
    print("  - light_purple/purple: Workflows, Rewards, OpenAI (RL-specific)")
    print("  - light_green: Stats, Perf, Dataset, Trainers (data/metrics)")
    print("  - light_cyan/cyan: Engines, Platforms, MCore (compute backends)")
    print("  - WARNING/ERROR: yellow/red (always override)")
    print("=" * 70)

    # Create loggers for different components
    test_loggers = [
        ("LocalScheduler", "Scheduler starting up..."),
        ("LocalLauncher", "Launcher initializing..."),
        ("[FSDPEngine Rank 0]", "Initializing FSDP..."),
        ("[MegatronEngine Rank 1]", "Loading model weights..."),
        ("RLVRWorkflow", "Starting episode 1..."),
        ("MultiTurnWorkflow", "Processing turn 3..."),
        ("TrainController", "Creating workers..."),
        ("RolloutController", "Starting rollout..."),
        ("SGLangWrapper", "Server ready on port 8000"),
        ("StatsLogger", "Logging metrics..."),
        ("Dataset", "Loading training data..."),
        ("GSM8KReward", "Computing rewards..."),
        ("PPOActor", "Running PPO forward pass..."),
        ("CUDAPlatform", "Detected 8 GPUs"),
        ("OpenAIClient", "Connecting to API..."),
        ("UnknownLogger", "This uses default white color"),
    ]

    for logger_name, message in test_loggers:
        logger = getLogger(logger_name)
        logger.info(message)

    print()
    print("Testing WARNING/ERROR override (should be yellow/red):")
    getLogger("LocalScheduler").warning("This warning should be yellow")
    getLogger("[FSDPEngine Rank 0]").error("This error should be red")
    getLogger("RLVRWorkflow").critical("This critical should be red bg")
