"""Check the actual inference RPC import chain without starting a server."""

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

modules = {}
for name in [
    "msgpack",
    "ray",
    "areal.infra.rpc.rpc_server",
    "areal.engine.sglang_remote",
    "areal.engine.awex.sglang_plugin",
    "areal.experimental.openai.proxy.proxy_rollout_server",
    "examples.swe.arena_agent",
]:
    module = importlib.import_module(name)
    modules[name] = str(module.__file__)
result = {
    "rpc_import_chain_verified": True,
    "proxy_entry_import_verified": True,
    "arena_agent_import_verified": True,
    "proxy_dependencies": {
        name: metadata.version(name) for name in ("litellm", "fastuuid")
    },
    "modules": modules,
    "sglang_version": metadata.version("sglang"),
    "ray_version": metadata.version("ray"),
    "no_server_or_model_instantiated": True,
}
assert result["sglang_version"] == "0.5.18.dev10+g85b539146"
for name in (
    "areal.experimental.openai.proxy.proxy_rollout_server",
    "examples.swe.arena_agent",
):
    assert Path(modules[name]).is_relative_to(Path(os.environ["QWEN_REPO"]))
assert result["proxy_dependencies"] == {"litellm": "1.83.7", "fastuuid": "0.14.0"}
(Path(os.environ["EXPERIMENT_ROOT"]) / "rollout-import-validation.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("ROLLOUT_IMPORT_PREFLIGHT_PASS " + json.dumps(result), flush=True)
