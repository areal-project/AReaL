# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import click

from areal.experimental.cli.client import ServiceHTTPError, ServiceUnreachable
from areal.experimental.cli.inference.client import GatewayClient
from areal.experimental.cli.inference.common import logger
from areal.experimental.cli.inference.lifecycle import inf_lifecycle


@click.command(
    name="collect",
    help="Start sessions, wait for ready trajectories, export and dump them.",
)
@click.option(
    "--model-name", "model", required=True, help="Model name to collect from."
)
@click.option("--service", default=None, help="Target service instance.")
@click.option(
    "--batch-size", type=int, required=True, help="Number of sessions to collect."
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--timeout", type=float, default=1800.0, show_default=True)
@click.option("--poll-interval", type=float, default=2.0, show_default=True)
@click.option("--turn-discount", type=float, default=1.0, show_default=True)
@click.option(
    "--export-style",
    type=click.Choice(["individual", "concat"]),
    default="individual",
    show_default=True,
)
@click.option(
    "--format",
    type=click.Choice(["json", "jsonl"]),
    default="jsonl",
    show_default=True,
)
@click.option("--json", "json_progress", is_flag=True, help="Emit progress events.")
def collect_cmd(
    model: str,
    service: str | None,
    batch_size: int,
    output: Path | None,
    timeout: float,
    poll_interval: float,
    turn_discount: float,
    export_style: str,
    format: str,
    json_progress: bool,
) -> None:
    raise SystemExit(
        do_collect(
            model=model,
            service=service,
            batch_size=batch_size,
            output=output,
            timeout=timeout,
            poll_interval=poll_interval,
            turn_discount=turn_discount,
            export_style=export_style,
            format=format,
            json_progress=json_progress,
        )
        or 0
    )


def do_collect(
    *,
    model: str,
    batch_size: int,
    output: Path | None,
    timeout: float,
    poll_interval: float,
    turn_discount: float,
    export_style: str,
    format: str,
    json_progress: bool,
    service: str | None = None,
) -> int:
    state = inf_lifecycle.load_running_state(service)
    gateway = GatewayClient(state.gateway_url, state.admin_api_key)

    event = _make_event_writer(json_progress)
    task_id = "cli-collect"
    event({"event": "starting", "model": model, "batch_size": batch_size})
    try:
        start = gateway.start_session(
            model=model,
            task_id=task_id,
            group_size=batch_size,
        )
    except (ServiceUnreachable, ServiceHTTPError) as exc:
        raise click.ClickException(f"start_session failed: {exc}") from exc

    group_id = start.get("group_id", "")
    sessions = start.get("sessions") or []
    session_ids = [sess["session_id"] for sess in sessions]
    for sess in sessions:
        event(
            {
                "event": "session_started",
                "session_id": sess.get("session_id"),
                "session_api_key": sess.get("session_api_key"),
            }
        )

    chunks: list[dict[str, Any]] = []
    collected = 0
    deadline = time.time() + timeout
    while collected < batch_size and time.time() < deadline:
        try:
            response = gateway.export_trajectories(
                session_ids=session_ids,
                remove_session=False,
                discount=turn_discount,
                style=export_style,
            )
        except ServiceHTTPError as exc:
            raise click.ClickException(f"export_trajectories failed: {exc}") from exc
        except ServiceUnreachable as exc:
            logger.warning("gateway unreachable mid-poll: %s", exc)
            time.sleep(poll_interval)
            continue

        traj = response.get("traj") or {}
        chunk_count = _estimate_trajectory_count(traj)
        if chunk_count > 0:
            collected += chunk_count
            chunks.append({"count": chunk_count, "traj": traj})
            event(
                {
                    "event": "exported",
                    "count": chunk_count,
                    "collected": collected,
                    "target": batch_size,
                }
            )
        else:
            event({"event": "waiting", "collected": collected, "target": batch_size})
            time.sleep(poll_interval)

    _cleanup_sessions(gateway, session_ids=session_ids, group_id=group_id)
    payload = {
        "model": model,
        "group_id": group_id,
        "sessions": sessions,
        "collected": collected,
        "target": batch_size,
        "chunks": chunks,
    }
    _write_payload(payload, output=output, format=format)
    if collected < batch_size:
        event(
            {
                "event": "timeout",
                "collected": collected,
                "target": batch_size,
                "timeout": timeout,
            }
        )
        return 1
    event({"event": "done", "collected": collected})
    return 0


def _make_event_writer(json_progress: bool):
    def emit(payload: dict[str, Any]) -> None:
        if json_progress:
            click.echo(json.dumps(payload))
            return
        event = payload.get("event", "event")
        if event == "session_started":
            click.echo(
                "session "
                f"{payload.get('session_id')} "
                f"api_key={payload.get('session_api_key')}",
                err=True,
            )
        elif event in {"exported", "waiting", "timeout", "done"}:
            click.echo(json.dumps(payload), err=True)

    return emit


def _estimate_trajectory_count(traj: dict[str, Any]) -> int:
    if not traj:
        return 0
    interactions = traj.get("interactions")
    if isinstance(interactions, list):
        return len(interactions)
    rewards = traj.get("rewards")
    shape = _serialized_shape(rewards)
    if shape:
        return int(shape[0])
    return 1


def _serialized_shape(value: Any) -> list[int] | None:
    if not isinstance(value, dict):
        return None
    if value.get("type") == "tensor":
        shape = value.get("shape")
        return shape if isinstance(shape, list) else None
    if value.get("type") == "dataclass":
        data = value.get("data")
        if isinstance(data, dict):
            return _serialized_shape(data.get("data"))
    return None


def _cleanup_sessions(
    gateway: GatewayClient,
    *,
    session_ids: list[str],
    group_id: str | None,
) -> None:
    if not session_ids:
        return
    try:
        gateway.export_trajectories(
            session_ids=session_ids,
            group_id=group_id,
            remove_session=True,
        )
    except (ServiceHTTPError, ServiceUnreachable) as exc:
        logger.warning("session cleanup failed: %s", exc)


def _write_payload(
    payload: dict[str, Any], *, output: Path | None, format: str
) -> None:
    if format == "json":
        text = json.dumps(payload, indent=2) + "\n"
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text)
        else:
            click.echo(text, nl=False)
        return

    lines = [
        json.dumps({"type": "session", **session})
        for session in payload.get("sessions", [])
    ]
    lines.extend(
        json.dumps({"type": "trajectory_chunk", **chunk})
        for chunk in payload.get("chunks", [])
    )
    text = "\n".join(lines) + ("\n" if lines else "")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        click.echo(text, nl=False)
