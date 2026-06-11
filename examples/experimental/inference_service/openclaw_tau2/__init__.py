import sys
import types

from loguru import logger

from .task_runner import run_task
from .task_runner_socket import run_task_with_socket_server
from .tau2_env import (
    EnvironmentSocketServer,
    OpenClawEnvironmentEvaluator,
    create_openclaw_tool_script,
    evaluate_simulation_with_environment,
)

__version__ = "0.2.0"


def register_openclaw_agent() -> bool:
    try:
        from tau2.registry import registry

        from .openclaw import OpenClawAgent

        registry.register_agent(OpenClawAgent, "openclaw_agent")
        logger.info("Registered OpenClaw agent with TAU²: 'openclaw_agent'")
        return True
    except ImportError as exc:
        logger.debug("TAU² not available, skipping registration: {}", exc)
    except ValueError as exc:
        if "already registered" in str(exc):
            logger.debug("OpenClaw agent already registered")
            return True
        logger.warning("Failed to register OpenClaw agent: {}", exc)
    except Exception as exc:
        logger.warning("Unexpected error registering OpenClaw agent: {}", exc)
    return False


def register_plugin() -> bool:
    return register_openclaw_agent()


_register_module = types.ModuleType(f"{__name__}.register")
_register_module.register_openclaw_agent = register_openclaw_agent
_register_module.register_plugin = register_plugin
sys.modules.setdefault(_register_module.__name__, _register_module)
_registration_success = register_openclaw_agent()
__all__ = [
    "run_task",
    "run_task_with_socket_server",
    "evaluate_simulation_with_environment",
    "OpenClawEnvironmentEvaluator",
    "EnvironmentSocketServer",
    "create_openclaw_tool_script",
]
