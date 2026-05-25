# SPDX-License-Identifier: Apache-2.0

"""``areal inf register`` — register a model with a running service.

Design §11.7. The CLI owns the full lifecycle of any backend processes it
launches for *internal* models (data proxies + SGLang/vLLM inference
servers). For *external* models, registration is a single ``POST
/register_model`` call.

Internal model launch involves spawning ``python -m
areal.experimental.inference_service.data_proxy`` and ``python -m
areal.experimental.inference_service.sglang.launch_server`` with allocated
ports and GPU plans derived from ``--backend <engine>:tp=...,pp=...,dp=...``.
That implementation is not yet wired up — the external path lands first
because it lets us exercise the full gateway lifecycle end-to-end without
GPUs or SGLang/vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from areal.experimental.cli.commands.inf._common import (
    add_targeting_flags,
    resolve_target,
)
from areal.experimental.cli.gateway_client import GatewayError
from areal.experimental.cli.inf_models import ModelEntry, ModelRegistry


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "register",
        help="Register a model with the running inference service.",
        description=(
            "Register an internal or external model with a running service. "
            "If --api-url is provided the model is external (remote API). "
            "Otherwise it is internal (local backend) — internal launch is "
            "not yet implemented in this CLI."
        ),
    )
    add_targeting_flags(p)
    p.add_argument("model_name", help="Unique logical model name.")
    p.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Free-form metadata stored in local state. Repeatable.",
    )
    # External-model flags
    p.add_argument(
        "--api-url",
        default=None,
        help="Base URL of the external OpenAI-compatible provider. "
        "Presence of this flag marks the model as external.",
    )
    p.add_argument(
        "--provider-api-key",
        default=None,
        help="Provider API key value (literal). Mutually exclusive with --provider-api-key-env.",
    )
    p.add_argument(
        "--provider-api-key-env",
        default=None,
        help="Environment variable name containing the provider API key.",
    )
    p.add_argument(
        "--provider-model",
        default=None,
        help="Remote model name to send upstream (defaults to model-name).",
    )
    # Internal-model flags (accepted but unimplemented; surface them in --help
    # so users see what's coming).
    p.add_argument("--backend", default=None, help="(internal) Backend spec, e.g. `sglang:tp=2,dp=1`.")
    p.add_argument("--model-path", default=None, help="(internal) HF/local model path.")
    p.add_argument("--tokenizer-path", default=None, help="(internal) Tokenizer path.")
    p.add_argument("--n-gpus-per-node", type=int, default=None, help="(internal) GPUs per node.")
    p.add_argument(
        "--timeout", type=float, default=30.0, help="Gateway request timeout (s)."
    )
    p.set_defaults(func=_handle)


def _parse_metadata(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--metadata expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    if args.provider_api_key and args.provider_api_key_env:
        raise SystemExit(
            "Pass at most one of --provider-api-key / --provider-api-key-env."
        )
    if args.provider_api_key:
        return args.provider_api_key
    if args.provider_api_key_env:
        key = os.environ.get(args.provider_api_key_env)
        if not key:
            raise SystemExit(
                f"Environment variable {args.provider_api_key_env!r} is empty or unset."
            )
        return key
    return None


def _handle(args: argparse.Namespace) -> int:
    target = resolve_target(args)
    client = target.client(timeout=args.timeout)

    is_external = args.api_url is not None
    if not is_external:
        raise NotImplementedError(
            "`areal inf register` for INTERNAL models (data proxy + "
            "SGLang/vLLM launch) is not yet implemented in this CLI. "
            "Pass --api-url <provider-base-url> to register an EXTERNAL model "
            "(OpenAI-compatible API). The internal path is tracked separately."
        )

    metadata = _parse_metadata(args.metadata)
    provider_model = args.provider_model or args.model_name
    api_key = _resolve_api_key(args)

    if target.service is None:
        raise SystemExit(
            "register requires a local service (so it can persist model state). "
            "Pass --service or start a service with `areal inf run`."
        )

    registry = ModelRegistry.load(target.service)
    if registry.get(args.model_name) is not None:
        raise SystemExit(
            f"Model {args.model_name!r} already registered on service "
            f"{target.service!r}. Deregister it first."
        )

    try:
        result = client.register_model(
            args.model_name,
            url=args.api_url,
            api_key=api_key,
            data_proxy_addrs=[],
        )
    except GatewayError as e:
        raise SystemExit(f"Gateway register_model failed: {e}") from e

    entry = ModelEntry(
        name=args.model_name,
        type="external",
        url=args.api_url,
        provider_model=provider_model,
        api_key_present=api_key is not None,
        created_at=time.time(),
        metadata=metadata,
    )
    registry.add(entry)
    registry.save()

    if args.json:
        print(json.dumps({
            "service": target.service,
            "model": args.model_name,
            "type": "external",
            "default": registry.default_model == args.model_name,
            "gateway_response": result,
        }, indent=2))
    else:
        flag = " (default)" if registry.default_model == args.model_name else ""
        print(f"Registered external model {args.model_name!r}{flag} on service {target.service!r}.")
    return 0
