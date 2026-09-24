# SPDX-License-Identifier: Apache-2.0
"""Qwen thinking defaults, overridden by explicit request options."""

from functools import wraps

THINKING_KEYS = {"enable_thinking", "thinking", "thinking_option"}
DEFAULTS = dict(enable_thinking=True, reasoning_effort="medium", thinking_option=None)


def with_template_defaults(extra_body=None):
    body = dict(extra_body or {})
    requested = dict(body.get("chat_template_kwargs") or {})
    # A request-level thinking switch overrides the whole default group.
    defaults = {
        key: value
        for key, value in DEFAULTS.items()
        if not (THINKING_KEYS.intersection(requested) and key in THINKING_KEYS)
    }
    body["chat_template_kwargs"] = {**defaults, **requested}
    return body


def wrap_create(original):
    @wraps(original)
    async def create(self, *args, **kwargs):
        kwargs["extra_body"] = with_template_defaults(kwargs.get("extra_body"))
        return await original(self, *args, **kwargs)

    return create
