# SPDX-License-Identifier: Apache-2.0

"""``areal inf collect`` — collect online trajectories from a registered model.

Design §11.9. The design composes ``collect`` client-side from four gateway
endpoints (``/grant_capacity``, ``/rl/start_session``, ``/rl/set_reward``,
``/export_trajectories``). At the time this command was first written the
gateway did **not** expose ``/grant_capacity`` — that endpoint lives on the
separate OpenAI proxy rollout server used by the training path, not on the
inference gateway.

Implementing a meaningful collect on top of just ``start_session`` +
``export_trajectories`` is brittle: sessions would have to be filled by
external agents calling ``chat/completions`` and ``set_reward`` while the CLI
polls, with no admission control and no guarantee that the gateway side
honors the request shape we'd need.

Rather than ship an orchestration that silently misbehaves, this command
intentionally errors out until the gateway grows a ``/grant_capacity``
endpoint (or the design specifies an alternate path). Users who need online
trajectory collection today should drive it through the training controller
(``RolloutController.rollout_batch``) instead.
"""

from __future__ import annotations

import argparse

from areal.experimental.cli.commands.inf._common import add_targeting_flags


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "collect",
        help="(planned) Collect online trajectories through the gateway.",
        description=(
            "Collect a batch of online trajectories. Not yet wired: the "
            "design's POST /grant_capacity endpoint is not exposed on the "
            "inference gateway. Use the training controller's rollout_batch "
            "path for online collection in the meantime."
        ),
    )
    add_targeting_flags(p)
    p.add_argument("model_name", nargs="?", default=None, help="Registered model (default: service default).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--turn-discount", type=float, default=1.0)
    p.add_argument("--export-style", default="individual")
    p.add_argument("--output", default=None)
    p.add_argument("--format", default="json", choices=["json", "jsonl"])
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "`areal inf collect` is not yet implemented. The inference gateway "
        "does not expose POST /grant_capacity (it lives on the OpenAI proxy "
        "rollout server used by training). Drive online collection through "
        "`RolloutController.rollout_batch` for now."
    )
