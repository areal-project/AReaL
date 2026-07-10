# SPDX-License-Identifier: Apache-2.0

"""AReaL Sync RPC Server — Guard + Data + Engine composition.

This module composes the shared Guard with data and engine blueprints
to create the full RPC server used by training workers.

Usage::

    python -m areal.infra.rpc.rpc_server \\
        --experiment-name exp1 --trial-name trial1 \\
        --role actor --worker-index 0
"""

from __future__ import annotations

import logging as stdlib_logging
import os
import sys


def _early_set_alloc_conf() -> None:
    """Set the CUDA allocator config by role before any CUDA-initializing import.

    The ``from areal...`` import chain below initializes CUDA at import time,
    which locks in the allocator config; setting ``PYTORCH_CUDA_ALLOC_CONF`` in
    ``main()`` is too late (expandable_segments would never take effect). So we
    pre-parse ``--role`` from argv and set the env var here, before any areal
    import. Only configs that explicitly set ``AWEX_ACTOR_ALLOC_CONF`` opt in
    (AWEX colocate); plain training runs are not implicitly affected. Inference
    roles (rollout/sglang) stay disabled: enabling expandable segments breaks
    SGLang engine initialization.
    """
    role = ""
    for i, a in enumerate(sys.argv):
        if a == "--role" and i + 1 < len(sys.argv):
            role = sys.argv[i + 1]
        elif a.startswith("--role="):
            role = a.split("=", 1)[1]
    is_inference = ("rollout" in role.lower()) or ("sglang" in role.lower())
    conf = os.environ.get("AWEX_ACTOR_ALLOC_CONF", "").strip()
    if role and not is_inference and conf:
        existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
        if existing:
            existing_keys = {
                part.split(":", 1)[0].split("=", 1)[0].strip()
                for part in existing.split(",")
                if part.strip()
            }
            extra_parts = []
            for part in conf.split(","):
                part = part.strip()
                key = part.split(":", 1)[0].split("=", 1)[0].strip()
                if part and key not in existing_keys:
                    extra_parts.append(part)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join([existing, *extra_parts])
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf


_early_set_alloc_conf()

from areal.infra.rpc.guard.app import (  # noqa: E402
    GuardState,
    configure_state_from_args,
    create_app,
    make_base_parser,
    run_server,
)
from areal.infra.rpc.guard.data_blueprint import data_bp  # noqa: E402
from areal.infra.rpc.guard.engine_blueprint import (  # noqa: E402
    engine_bp,
    register_engine_hooks,
)
from areal.utils import logging, perf_tracer  # noqa: E402

logger = logging.getLogger("SyncRPCServer")


def main():
    parser = make_base_parser(
        description="AReaL Sync RPC Server for TrainEngine/InferenceEngine"
    )
    parser.add_argument(
        "--werkzeug-log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for Werkzeug (Flask's WSGI server). Default: WARNING",
    )

    args, _ = parser.parse_known_args()

    werkzeug_logger = stdlib_logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(getattr(stdlib_logging, args.werkzeug_log_level))

    state = GuardState()
    bind_host = configure_state_from_args(state, args)

    app = create_app(state)
    app.register_blueprint(data_bp)
    app.register_blueprint(engine_bp)
    register_engine_hooks(state)

    state.register_cleanup_hook(lambda: perf_tracer.save(force=True))

    logger.info(f"Werkzeug log level: {args.werkzeug_log_level}")

    run_server(state, app, bind_host, args.port)


if __name__ == "__main__":
    main()
