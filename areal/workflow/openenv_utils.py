# SPDX-License-Identifier: Apache-2.0
"""Default action parsers and observation formatters for OpenEnvWorkflow.

Environment authors typically ship their own ``EnvClient`` subclass and Action
dataclass; these helpers cover the common cases so most environments plug in
with zero user code.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any

from areal.api.openenv_api import ActionParser, ObservationFormatter
from areal.utils import logging

logger = logging.getLogger("OpenEnvWorkflow")


def _import_from_string(path: str) -> Any:
    """Import ``module.submodule:attr`` or ``module.submodule.attr``."""
    if ":" in path:
        module_path, _, attr = path.rpartition(":")
    else:
        module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Import path {path!r} must include a module (e.g. 'pkg.mod.Cls')."
        )
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"Module {module_path!r} has no attribute {attr!r}.") from e


def _observation_to_text(observation: Any) -> str:
    """Best-effort stringification of an OpenEnv observation payload.

    Handles dataclasses (via ``dataclasses.asdict``), plain dicts, strings, and
    objects with ``.__dict__``. Falls back to ``repr``.
    """
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if is_dataclass(observation) and not isinstance(observation, type):
        try:
            return json.dumps(dataclasses.asdict(observation), ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    if isinstance(observation, dict):
        try:
            return json.dumps(observation, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    if hasattr(observation, "__dict__"):
        try:
            return json.dumps(
                {k: v for k, v in vars(observation).items() if not k.startswith("_")},
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            pass
    return repr(observation)


class AutoObservationFormatter:
    """Default formatter: dump the observation as JSON in a user message."""

    def __init__(self, prefix: str = "Observation:\n") -> None:
        self.prefix = prefix

    def __call__(self, observation: Any, step: int) -> dict[str, str]:
        return {
            "role": "user",
            "content": f"{self.prefix}{_observation_to_text(observation)}",
        }


class JSONActionParser:
    """Parse the last JSON object in the completion into a dict.

    Compatible with LLM outputs that mix reasoning text with a JSON action
    block. Returns ``None`` when no valid JSON object can be extracted.
    """

    _pattern = re.compile(r"\{[^{}]*\}|\{.*\}", re.DOTALL)

    def __call__(self, completion: str, observation: Any) -> dict | None:
        stripped = completion.strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
        matches = self._pattern.findall(completion)
        for candidate in reversed(matches):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
        return None


class TagActionParser:
    """Parse a ``<action>...</action>`` (or user-specified tag) payload.

    Body is parsed as JSON when it looks like an object/array, else returned
    as a stripped string. Returns ``None`` when the tag is absent.
    """

    def __init__(self, tag: str = "action") -> None:
        self.tag = tag
        self._pattern = re.compile(
            rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE
        )

    def __call__(self, completion: str, observation: Any) -> Any:
        match = self._pattern.search(completion)
        if match is None:
            return None
        body = match.group(1).strip()
        if body.startswith(("{", "[")):
            try:
                return json.loads(body)
            except (json.JSONDecodeError, ValueError):
                return body
        return body


class PassthroughActionParser:
    """Return the trimmed completion text as the action."""

    def __call__(self, completion: str, observation: Any) -> str:
        return completion.strip()


_ACTION_PARSER_ALIASES: dict[str, Callable[[], ActionParser]] = {
    "json": JSONActionParser,
    "tag": TagActionParser,
    "passthrough": PassthroughActionParser,
}

_OBS_FORMATTER_ALIASES: dict[str, Callable[[], ObservationFormatter]] = {
    "auto": AutoObservationFormatter,
}


def resolve_action_parser(spec: str) -> ActionParser:
    """Instantiate an :class:`ActionParser` from a shorthand or import path."""
    factory = _ACTION_PARSER_ALIASES.get(spec)
    if factory is not None:
        return factory()
    obj = _import_from_string(spec)
    return obj() if isinstance(obj, type) else obj


def resolve_obs_formatter(spec: str) -> ObservationFormatter:
    """Instantiate an :class:`ObservationFormatter` from a shorthand or path."""
    factory = _OBS_FORMATTER_ALIASES.get(spec)
    if factory is not None:
        return factory()
    obj = _import_from_string(spec)
    return obj() if isinstance(obj, type) else obj


def build_action(raw_action: Any, action_class_path: str | None) -> Any:
    """Coerce a parser output into the concrete OpenEnv Action type.

    When ``action_class_path`` is set and the parser returned a ``dict``, the
    dict is expanded as ``ActionClass(**dict)``. Any other combination is
    passed through unchanged, matching what most ``EnvClient.step`` variants
    accept (dataclass instance, plain dict, or primitive).
    """
    if action_class_path is None:
        return raw_action
    action_cls = _import_from_string(action_class_path)
    if isinstance(raw_action, dict):
        try:
            return action_cls(**raw_action)
        except TypeError as e:
            logger.warning(
                f"Failed to instantiate {action_class_path} from parsed action "
                f"{raw_action!r}: {e}. Falling back to passthrough."
            )
            return raw_action
    return raw_action


__all__ = [
    "AutoObservationFormatter",
    "JSONActionParser",
    "PassthroughActionParser",
    "TagActionParser",
    "_import_from_string",
    "build_action",
    "resolve_action_parser",
    "resolve_obs_formatter",
]
