# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from areal.api.cli_args import PRMConfig


@dataclass
class DataProxyConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    backend_addr: str = "http://localhost:30000"  # co-located SGLang/vLLM
    backend_type: str = "sglang"
    tokenizer_path: str = ""
    log_level: str = "warning"
    request_timeout: float = 120.0  # seconds per SGLang call
    set_reward_finish_timeout: float = 0.0
    max_resubmit_retries: int = 20  # max abort/resubmit cycles before giving up
    resubmit_wait: float = 0.5  # seconds between is_paused polls
    admin_api_key: str = "areal-admin-key"  # admin key for authentication
    callback_server_addr: str = ""
    deterministic_sampling: bool = False
    # Resolved serving address (host:port) used as node_addr for RTensor shards.
    # Set at startup by __main__.py after the host is resolved.
    serving_addr: str = ""

    # ArealOpenAI client parameters (forwarded from AgentConfig)
    tool_call_parser: str = "qwen"
    reasoning_parser: str = "qwen3"
    engine_max_tokens: int | None = None
    chat_template_type: str = "hf"
    message_preprocessors: tuple[str, ...] = ()
    prefix_matcher: str | None = None
    prm: PRMConfig = field(default_factory=PRMConfig)

    def __post_init__(self) -> None:
        if self.prm.enabled and self.prm.scorers and self.prm.error_policy != "reject":
            raise ValueError(
                "PRM keep_original error policy is only supported by the v1 proxy"
            )
        if (
            self.prm.enabled
            and self.prm.scorers
            and self.chat_template_type != "concat"
        ):
            raise ValueError("PRM scorers require chat_template_type='concat'")
