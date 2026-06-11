import sys
import types

from .agent import OpenClawAgent
from .service import OpenClawConfig, OpenClawService
from .workspace_manager import OpenClawWorkspaceManager

_config_module = types.ModuleType(f"{__name__}.config")
_config_module.OpenClawConfig = OpenClawConfig
sys.modules.setdefault(_config_module.__name__, _config_module)
__all__ = [
    "OpenClawAgent",
    "OpenClawService",
    "OpenClawWorkspaceManager",
    "OpenClawConfig",
]
