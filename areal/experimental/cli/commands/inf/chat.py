# SPDX-License-Identifier: Apache-2.0

"""``areal inf chat`` — chat with a registered model.

Design §11.10. Supports a single-shot prompt (positional) and a REPL when
no prompt is given. Streaming is the default. Implemented entirely against
the gateway's ``POST /chat/completions`` endpoint, so this command works
both with local services launched by ``areal inf run`` and with arbitrary
remote gateways via ``--gateway-url``.

The REPL supports a minimal set of slash commands so the chat surface is
useful before the full design §8.3 shell is implemented:

    /bye, /exit, /quit       Leave the REPL.
    /clear                   Drop conversation history.
    /system <text>           Set / replace the system message.
    /model                   Show the active model.
    /model <name>            Switch the active model (no validation).
"""

from __future__ import annotations

import argparse
import json
import sys

from areal.experimental.cli.commands.inf._common import (
    add_targeting_flags,
    resolve_target,
)
from areal.experimental.cli.gateway_client import GatewayClient, GatewayError
from areal.experimental.cli.inf_models import ModelRegistry, resolve_model_name


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "chat",
        help="Chat with a registered model.",
        description=(
            "Send a single prompt or open a REPL against a registered model. "
            "If --model is omitted, falls back to the service's default model. "
            "If no positional prompt is given, enters interactive mode."
        ),
    )
    add_targeting_flags(p)
    p.add_argument(
        "--model",
        default=None,
        help="Registered model name (default: service's default model).",
    )
    p.add_argument(
        "prompt",
        nargs="*",
        help="Single-shot prompt (joined with spaces). Omit to enter REPL.",
    )
    p.add_argument("--system", default=None, help="Optional system message.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter.")
    p.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Maximum completion tokens.",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (wait for the full response).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request socket timeout (s).",
    )
    p.set_defaults(func=_handle)


def _gen_kwargs(args: argparse.Namespace) -> dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_tokens": args.max_completion_tokens,
    }


def _single_shot(
    client: GatewayClient,
    *,
    model: str,
    messages: list[dict],
    stream: bool,
    args: argparse.Namespace,
) -> int:
    try:
        result = client.chat_completion(
            model=model, messages=messages, stream=stream, **_gen_kwargs(args)
        )
    except GatewayError as e:
        raise SystemExit(f"chat failed: {e}") from e

    if stream:
        any_output = False
        for piece in result:
            sys.stdout.write(piece)
            sys.stdout.flush()
            any_output = True
        if any_output:
            sys.stdout.write("\n")
        return 0

    # Non-streaming
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    choices = (result or {}).get("choices") or []
    if not choices:
        print(json.dumps(result, default=str))
        return 0
    msg = (choices[0].get("message") or {}).get("content", "")
    print(msg)
    return 0


def _repl(
    client: GatewayClient,
    *,
    initial_model: str,
    system: str | None,
    stream: bool,
    args: argparse.Namespace,
) -> int:
    model = initial_model
    history: list[dict] = []
    if system:
        history.append({"role": "system", "content": system})

    print(
        f"areal inf chat REPL — model={model!r}. /bye to exit, /clear to reset, "
        f"/system <text>, /model [name]."
    )

    while True:
        try:
            line = input(">>> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 130
        text = line.strip()
        if not text:
            continue
        if text in ("/bye", "/exit", "/quit"):
            return 0
        if text == "/clear":
            sys_msg = history[0] if history and history[0]["role"] == "system" else None
            history = [sys_msg] if sys_msg else []
            print("(history cleared)")
            continue
        if text.startswith("/system"):
            new_sys = text[len("/system") :].strip()
            history = [h for h in history if h["role"] != "system"]
            if new_sys:
                history.insert(0, {"role": "system", "content": new_sys})
                print(f"(system message set to {new_sys!r})")
            else:
                print("(system message removed)")
            continue
        if text == "/model":
            print(f"current model: {model!r}")
            continue
        if text.startswith("/model "):
            model = text[len("/model ") :].strip()
            print(f"(switched to {model!r})")
            continue
        if text.startswith("/"):
            print(f"(unknown command {text!r})")
            continue

        history.append({"role": "user", "content": text})
        try:
            result = client.chat_completion(
                model=model, messages=history, stream=stream, **_gen_kwargs(args)
            )
        except GatewayError as e:
            print(f"(chat error: {e})")
            history.pop()
            continue
        if stream:
            buf: list[str] = []
            for piece in result:
                sys.stdout.write(piece)
                sys.stdout.flush()
                buf.append(piece)
            sys.stdout.write("\n")
            assistant_text = "".join(buf)
        else:
            choices = (result or {}).get("choices") or []
            assistant_text = (
                (choices[0].get("message") or {}).get("content", "")
                if choices else ""
            )
            print(assistant_text)
        history.append({"role": "assistant", "content": assistant_text})


def _handle(args: argparse.Namespace) -> int:
    target = resolve_target(args)
    client = target.client(timeout=args.timeout)

    if args.model is None:
        if target.service is None:
            raise SystemExit(
                "chat needs either --model or a local service (to find the default model)."
            )
        model = resolve_model_name(target.service, None)
    else:
        model = args.model
        if target.service is not None:
            reg = ModelRegistry.load(target.service)
            if reg.get(model) is None:
                # Not fatal — the gateway may know about models the CLI didn't
                # register, e.g. via --gateway-url to a remote service.
                pass

    stream = not args.no_stream

    if args.prompt:
        prompt = " ".join(args.prompt)
        messages: list[dict] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        return _single_shot(
            client, model=model, messages=messages, stream=stream, args=args
        )

    return _repl(
        client,
        initial_model=model,
        system=args.system,
        stream=stream,
        args=args,
    )
