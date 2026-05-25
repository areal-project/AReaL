# SPDX-License-Identifier: Apache-2.0

"""Per-service local model registry under ``~/.areal/inf/models/<service>.json``.

This file is the CLI's source of truth for which models a service has
registered — what *type* (internal vs. external), what backend processes
(if any) the CLI launched on the user's behalf, and which model is the
*default* (used by ``chat`` / ``collect`` when the user omits ``model-name``).

Design §9.2. The gateway has no concept of a "default" model; that lives
purely in this file.

We never store provider API keys in this file. Internal-model backend
addresses (data proxies, inference servers) ARE stored so that ``deregister``
can clean them up later.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.inf_state import inf_dir


def models_dir() -> Path:
    d = inf_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_file(service: str) -> Path:
    return models_dir() / f"{service}.json"


@dataclass
class ModelEntry:
    name: str
    type: str  # "internal" | "external"
    url: str = ""
    provider_model: str = ""
    api_key_present: bool = False
    dp_addrs: list[str] = field(default_factory=list)
    inference_pids: list[int] = field(default_factory=list)
    data_proxy_pids: list[int] = field(default_factory=list)
    backend: str = ""  # e.g. "sglang:tp=2,dp=1" (internal only)
    model_path: str = ""  # internal only
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelRegistry:
    service: str
    default_model: str | None = None
    models: list[ModelEntry] = field(default_factory=list)

    @classmethod
    def load(cls, service: str) -> ModelRegistry:
        p = models_file(service)
        if not p.exists():
            return cls(service=service, default_model=None, models=[])
        with open(p) as f:
            raw = json.load(f)
        return cls(
            service=raw.get("service", service),
            default_model=raw.get("default_model"),
            models=[ModelEntry(**m) for m in raw.get("models", [])],
        )

    def save(self) -> None:
        p = models_file(self.service)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        doc = {
            "service": self.service,
            "default_model": self.default_model,
            "models": [asdict(m) for m in self.models],
        }
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
        os.replace(tmp, p)

    def get(self, name: str) -> ModelEntry | None:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def add(self, entry: ModelEntry) -> None:
        if self.get(entry.name) is not None:
            raise ValueError(
                f"Model {entry.name!r} already registered on service {self.service!r}."
            )
        if not entry.created_at:
            entry.created_at = time.time()
        self.models.append(entry)
        if self.default_model is None:
            self.default_model = entry.name

    def remove(self, name: str) -> ModelEntry | None:
        for i, m in enumerate(self.models):
            if m.name == name:
                removed = self.models.pop(i)
                if self.default_model == name:
                    self.default_model = self.models[0].name if self.models else None
                return removed
        return None

    def names(self) -> list[str]:
        return [m.name for m in self.models]


def resolve_model_name(service: str, explicit: str | None) -> str:
    """Apply design rule: explicit arg > default model > sole known > error."""
    reg = ModelRegistry.load(service)
    if explicit:
        if reg.get(explicit) is None:
            # Not in local state — caller may still attempt (e.g. the gateway
            # could have models the CLI did not register). Return the explicit
            # name unchanged.
            return explicit
        return explicit
    if reg.default_model:
        return reg.default_model
    if len(reg.models) == 1:
        return reg.models[0].name
    if not reg.models:
        raise SystemExit(
            f"No models registered on service {service!r}. "
            f"Add one with `areal inf register`."
        )
    raise SystemExit(
        f"Multiple models on service {service!r}; specify one. "
        f"Known: {', '.join(reg.names())}."
    )
