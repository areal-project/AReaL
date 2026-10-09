# SPDX-License-Identifier: Apache-2.0
"""Apply Qwen thinking defaults to Arena requests."""

from examples.swe.qwen38_flash_next.template_defaults import wrap_create


def main():
    from areal.experimental.openai.client import AsyncCompletionsWithReward
    from areal.experimental.openai.proxy import proxy_rollout_server

    AsyncCompletionsWithReward.create = wrap_create(AsyncCompletionsWithReward.create)
    proxy_rollout_server.main()


if __name__ == "__main__":
    main()
