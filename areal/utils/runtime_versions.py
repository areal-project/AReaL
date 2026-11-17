# SPDX-License-Identifier: Apache-2.0
"""Record loaded tokenizer and backend identities during initialization."""

import hashlib
import json
import os
import socket
import sys
from dataclasses import asdict
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from areal.utils import logging

logger = logging.getLogger("RuntimeVersions")
LOG_MARKER = "Runtime identity "


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def tokenizer_identity(tokenizer) -> dict:
    """Hash the loaded fast tokenizer state and effective default template."""
    try:
        template = tokenizer.get_chat_template()
        state = {
            "class": type(tokenizer).__name__,
            "backend": json.loads(tokenizer.backend_tokenizer.to_str()),
            "special_tokens": tokenizer.special_tokens_map,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
        }
    except (AttributeError, ValueError, TypeError) as error:
        return {"error": str(error)}
    return {
        "path": str(tokenizer.name_or_path),
        "sha256": _digest(state),
        "template_sha256": _digest(template),
    }


def record_tokenizer(tokenizer) -> None:
    logger.info(
        "%s%s",
        LOG_MARKER,
        json.dumps(
            {
                "role": "tokenizer",
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "tokenizer": tokenizer_identity(tokenizer),
            },
            sort_keys=True,
        ),
    )


@lru_cache(maxsize=16)
def _source_digest(origin: str, package: bool) -> str:
    path = Path(origin)
    paths = [path]
    if package:
        paths = sorted(
            item
            for item in path.parent.rglob("*")
            if item.is_file() and item.suffix in (".py", ".so")
        )
    digest = hashlib.sha256()
    for item in paths:
        digest.update(str(item.relative_to(path.parent)).encode() + b"\0")
        with item.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def math_verifier_identity(worker, reward_fn) -> dict:
    """Fingerprint the loaded BOBA verifier and its effective parameters."""
    identity = {"implementations": {}, "modules": {}, "parameters": {}}
    sources = {}
    try:
        identity["parameters"] = {
            "precision": worker.precision,
            "timeout": worker.timeout,
            "gold_extraction_target": [
                asdict(x) for x in worker.gold_extraction_target
            ],
            "pred_extraction_target": [
                asdict(x) for x in worker.pred_extraction_target
            ],
        }
        for label, implementation in (("reward", reward_fn), ("worker", type(worker))):
            name = implementation.__module__
            origin = str(Path(sys.modules[name].__file__).resolve())
            sources[label] = origin
            identity["implementations"][label] = {
                "name": name + "." + implementation.__qualname__,
                "source_sha256": _source_digest(origin, False),
            }
        for name, distribution in (
            ("math_verify", "math-verify"),
            ("latex2sympy2_extended", "latex2sympy2-extended"),
            ("sympy", "sympy"),
            ("antlr4", "antlr4-python3-runtime"),
        ):
            module = sys.modules.get(name)
            if module is None or not getattr(module, "__file__", None):
                raise ValueError(f"Verifier dependency is not loaded: {name}")
            origin = str(Path(module.__file__).resolve())
            sources[name] = origin
            try:
                installed = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                installed = None
            identity["modules"][name] = {
                "distribution_version": installed,
                "source_sha256": _source_digest(origin, True),
            }
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        return {"identity": identity, "sources": sources, "error": str(error)}
    return {"sha256": _digest(identity), "identity": identity, "sources": sources}


def record_runtime_versions(role: str, **identity) -> None:
    """Use loaded modules so PYTHONPATH overrides and patched sources are visible."""
    record = {
        "schema_version": 1,
        "role": role,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "executable": sys.executable,
        "modules": {},
        **identity,
    }
    try:
        for name, distribution in (
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("megatron.core", "megatron-core"),
            ("megatron.bridge", "megatron-bridge"),
            ("mbridge", "mbridge"),
            ("awex", "awex"),
            ("areal.engine.awex", "areal"),
            ("sglang", "sglang"),
        ):
            module = sys.modules.get(name)
            if module is None or not getattr(module, "__file__", None):
                continue
            origin = str(Path(module.__file__).resolve())
            try:
                installed = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                installed = None
            version = getattr(module, "__version__", None)
            record["modules"][name] = {
                "path": origin,
                "version": str(version) if version is not None else None,
                "distribution_version": installed,
                "source_sha256": _source_digest(
                    origin,
                    name
                    in (
                        "megatron.core",
                        "megatron.bridge",
                        "mbridge",
                        "awex",
                        "areal.engine.awex",
                        "sglang",
                    ),
                ),
            }
        torch = sys.modules.get("torch")
        record["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    except (OSError, ValueError) as error:
        record["error"] = str(error)
    logger.info("%s%s", LOG_MARKER, json.dumps(record, sort_keys=True))
