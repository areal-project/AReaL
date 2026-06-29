# SPDX-License-Identifier: Apache-2.0

"""TOML config loader for subcommand CLIs.

Each CLI's config file lives at ``$AREAL_HOME/<namespace>/config.toml``.
Sections + keys map to ``(verb, click_option_name)`` via a per-CLI
``bindings`` dict that subclasses provide; this base handles all the
flatten + merge + dispatch plumbing.

Precedence: explicit CLI flag > extra config passed via ``--config`` >
``$AREAL_HOME/<namespace>/config.toml`` > the click option's hard-coded
default.
"""

from __future__ import annotations

from pathlib import Path

from areal.experimental.cli.state import NamespacedStateStore

# (section, key) → (verb_name_or_tuple, click_option_name)
BindingMap = dict[tuple[str, str], tuple[str | tuple[str, ...], str]]


class ConfigLoader:
    """Subclass and set ``namespace`` + ``bindings`` (or pass them in)."""

    namespace: str = ""
    bindings: BindingMap = {}

    def __init__(
        self,
        *,
        namespace: str | None = None,
        bindings: BindingMap | None = None,
    ) -> None:
        if namespace is not None:
            self.namespace = namespace
        if bindings is not None:
            self.bindings = bindings
        if not self.namespace:
            raise ValueError("ConfigLoader requires a namespace")
        self.store = NamespacedStateStore(self.namespace)

    def default_config_path(self) -> Path:
        return self.store.config_path()

    def load_click_default_map(self, extra: Path | None = None) -> dict:
        """Return a ``click`` ``default_map`` built by merging the
        namespace's default config.toml with an optional extra file.

        ``extra`` (typically the ``--config FILE`` flag) wins on
        conflicts; bindings absent from the map are silently ignored.
        """

        merged: dict[tuple[str, str], object] = {}
        for path in (self.default_config_path(), extra):
            if path is None:
                continue
            merged.update(self._flatten(self._read_toml(path)))

        default_map: dict[str, dict] = {}
        for (section, key), value in merged.items():
            binding = self.bindings.get((section, key))
            if binding is None:
                continue
            verbs, opt = binding
            if isinstance(verbs, str):
                verbs = (verbs,)
            for verb in verbs:
                default_map.setdefault(verb, {})[opt] = value
        return default_map

    # ------------------------------------------------------------------
    # Helpers — subclasses generally don't need to override these

    def _read_toml(self, path: Path) -> dict:
        try:
            import tomllib
        except ImportError:
            return {}
        if not path.exists():
            return {}
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    def _flatten(self, toml: dict, parent: str = "") -> dict[tuple[str, str], object]:
        """Flatten nested TOML tables to ``(section, key) -> value``.

        Nested keys are dotted on the section side:
        ``[register.internal] backend = "..."`` →
        ``("register.internal", "backend") -> "..."``.
        """

        out: dict[tuple[str, str], object] = {}
        for k, v in toml.items():
            if isinstance(v, dict):
                sub_parent = f"{parent}.{k}" if parent else k
                for (sec, key), val in self._flatten(v, sub_parent).items():
                    out[(sec, key)] = val
            else:
                out[(parent, k)] = v
        return out
