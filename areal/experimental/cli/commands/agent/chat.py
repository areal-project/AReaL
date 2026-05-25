# SPDX-License-Identifier: Apache-2.0

"""``areal agent chat`` — single-turn or REPL chat with the agent service.

Uses the agent gateway's ``POST /v1/responses`` bridge (the
OpenResponsesBridge). The ``user`` field carries the session key, which is
how the data_proxy preserves multi-turn history. Streaming is intentionally
not supported here: the REST bridge aggregates events into a single
response. Token-level streaming would require WebSocket support, which the
thin urllib client deliberately avoids.

Optional transcript saving (default ON) writes one JSONL line per turn to
``~/.areal/agent/chats/<service>/<session>.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import time

from areal.experimental.cli.agent_gateway_client import (
    AgentGatewayError,
    AgentGatewayUnreachable,
    extract_response_text,
)
from areal.experimental.cli.agent_sessions import (
    SessionRegistry,
    resolve_session_key,
)
from areal.experimental.cli.agent_state import (
    agent_chats_dir,
)
from areal.experimental.cli.commands.agent._common import (
    add_targeting_flags,
    resolve_agent_target,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "chat",
        help="Single-shot or REPL chat with the agent in a session.",
        description=(
            "Send a prompt to the agent service via POST /v1/responses. "
            "If no prompt is given, enters a simple REPL. Session key "
            "resolves: explicit positional > default > sole-active > error."
        ),
    )
    add_targeting_flags(p)
    p.add_argument(
        "session_key",
        nargs="?",
        default=None,
        help="Target session key (defaults to current default session).",
    )
    p.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Single-shot prompt. Omit to enter REPL mode.",
    )
    p.add_argument(
        "--model",
        default="",
        help="Inference model name to pass through to the bridge.",
    )
    p.add_argument(
        "--instructions",
        default="",
        help="System/instructions prefix forwarded with each turn.",
    )
    p.add_argument(
        "--no-save-history",
        action="store_true",
        help="Disable JSONL transcript saving.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-turn request timeout (s).",
    )
    p.set_defaults(func=_handle)


def _append_history(path, role: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "role": role,
            "content": content,
        }) + "\n")


def _send_turn(
    target,
    session_key: str,
    message: str,
    *,
    model: str,
    instructions: str,
    timeout: float,
) -> tuple[str, dict | None, str | None]:
    client = target.client(timeout=timeout)
    try:
        resp = client.responses(
            session_key=session_key,
            message=message,
            model=model,
            instructions=instructions,
        )
    except AgentGatewayUnreachable as e:
        return "", None, f"agent gateway unreachable: {e}"
    except AgentGatewayError as e:
        return "", None, str(e)
    text = extract_response_text(resp)
    return text, resp, None


def _touch_session(service: str | None, session_key: str) -> None:
    if not service:
        return
    try:
        reg = SessionRegistry.load(service)
        entry = reg.get(session_key)
        if entry is not None:
            entry.touch()
            reg.save()
    except Exception:
        pass


def _handle(args: argparse.Namespace) -> int:
    target = resolve_agent_target(args)
    service = target.service or "default"
    session_key = resolve_session_key(service, args.session_key) if target.service else (
        args.session_key
        or _abort_remote_needs_session()
    )

    save_history = not args.no_save_history
    history_path = (
        agent_chats_dir(target.service) / f"{session_key}.jsonl"
        if target.service and save_history
        else None
    )

    if args.prompt is not None:
        if save_history and history_path is not None:
            _append_history(history_path, "user", args.prompt)
        text, resp, err = _send_turn(
            target,
            session_key,
            args.prompt,
            model=args.model,
            instructions=args.instructions,
            timeout=args.timeout,
        )
        if err:
            print(f"error: {err}")
            return 1
        if args.json:
            print(json.dumps({"session": session_key, "response": resp, "text": text}, indent=2))
        else:
            print(text)
        if save_history and history_path is not None:
            _append_history(history_path, "assistant", text)
        _touch_session(target.service, session_key)
        return 0

    print(
        f"Agent chat — service={service}, session={session_key} "
        f"(empty line or /exit to quit)"
    )
    while True:
        try:
            line = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        msg = line.strip()
        if not msg:
            continue
        if msg in ("/exit", "/quit", "/bye"):
            return 0
        if save_history and history_path is not None:
            _append_history(history_path, "user", msg)
        text, resp, err = _send_turn(
            target,
            session_key,
            msg,
            model=args.model,
            instructions=args.instructions,
            timeout=args.timeout,
        )
        if err:
            print(f"error: {err}")
            continue
        print(text)
        if save_history and history_path is not None:
            _append_history(history_path, "assistant", text)
        _touch_session(target.service, session_key)


def _abort_remote_needs_session() -> str:
    raise SystemExit(
        "Remote gateway mode (--gateway-url) requires an explicit session key."
    )
