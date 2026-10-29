from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class OSWorldAgentConfig(GRPOConfig):
    n_trajs: int = field(default=1)
    max_steps: int = field(default=15)
    max_workers: int = field(default=4)
    sleep_after_execution: float = field(default=1.0)
    env_reset_wait_secs: float = field(default=60.0)

    provider_name: str = field(default="docker")
    path_to_vm: str | None = field(default=None)
    os_type: str = field(default="Ubuntu")
    headless: bool = field(default=True)
    screen_width: int = field(default=1920)
    screen_height: int = field(default=1080)
    observation_type: str = field(default="screenshot")
    action_space: str = field(default="pyautogui")

    osworld_root: str = field(default="")
    evaluation_examples_dir: str = field(default="")
    test_meta_path: str = field(default="")
    osworld_cache_dir: str = field(default="cache")
    turn_discount: float = field(default=0.9)

    # When non-empty, the workflow skips the in-process `DesktopEnv` and
    # proxies reset/step/evaluate/close to a `remote_server.py` running on
    # a host that actually has docker available. Example: "http://10.0.0.5:8000".
    remote_server_url: str = field(default="")
    remote_request_timeout_secs: float = field(default=1800.0)

    # Remote sandbox cluster behind an HTTP/HTTPS gateway (preferred when
    # available). Setting both fields routes the workflow through
    # `gateway_sandbox.py`, which talks to the cluster endpoint via a
    # vendor-provided SDK (see `gateway_sandbox.py` for the expected protocol)
    # and OSWorld's gateway-proxied HTTP endpoints.
    gateway_endpoint: str = field(default="")
    gateway_token: str = field(default="")
    gateway_timeout_secs: int = field(default=1800)

    # Smoke-only ablation: drop image content from user turns and feed the
    # agent text-only stubs instead. Lets us point `actor.path` at a text-only
    # base model (e.g. Qwen3-4B-Instruct) and exercise the full PPO loop,
    # without needing the VL pipeline (`mm_token_type_ids`, etc.) wired up.
    # The agent operates "blind" — useful only for plumbing verification.
    text_only: bool = field(default=False)
