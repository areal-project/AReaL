---
package: sglang
github: sgl-project/sglang
branch_template: v${VERSION}
upstream_paths:
  - python/sglang/srt/entrypoints/engine.py
  - python/sglang/srt/entrypoints/http_server.py
  - python/sglang/srt/entrypoints/openai/protocol.py
  - python/sglang/srt/function_call/function_call_parser.py
  - python/sglang/srt/function_call/core_types.py
  - python/sglang/srt/managers/scheduler.py
  - python/sglang/srt/managers/io_struct.py
  - python/sglang/srt/parser/reasoning_parser.py
  - python/sglang/srt/server_args.py
  - python/sglang/srt/utils/__init__.py
  - python/sglang/srt/utils/common.py
  - python/sglang/srt/utils/network.py
  - python/sglang/srt/utils/numa_utils.py
  - python/sglang/launch_server.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                                           | Imports / Usage                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/sglang_remote.py`                                | HTTP-only (no Python imports). Sends requests to `/generate`, `/load_lora_adapter`, `/update_weights_from_disk`, `/update_weights_from_distributed`, `/init_weights_update_group`, `/pause_generation`, `/continue_generation`, `/health`, `/release_memory_occupation`, `/resume_memory_occupation`. Parses `meta_info`, `finish_reason`, `output_token_logprobs`, `routed_experts` from JSON responses.                                                                                                                   |
| `areal/experimental/openai/tool_call_parser.py`                | `sglang.srt.entrypoints.openai.protocol.Function`, `sglang.srt.entrypoints.openai.protocol.Tool`, `sglang.srt.function_call.function_call_parser.FunctionCallParser`, `sglang.srt.parser.reasoning_parser.ReasoningParser`                                                                                                                                                                                                                                                                                                  |
| `areal/experimental/inference_service/sglang/launch_server.py` | `sglang.srt.entrypoints.engine.Engine`, `sglang.srt.entrypoints.engine.init_tokenizer_manager`, `sglang.srt.entrypoints.http_server._execute_server_warmup`, `sglang.srt.entrypoints.http_server._setup_and_run_http_server`, `sglang.srt.entrypoints.http_server.app`, `sglang.srt.managers.detokenizer_manager.run_detokenizer_process`                                                                                                                                                                                   |
| `areal/experimental/inference_service/sglang/scheduler.py`     | `sglang.srt.managers.scheduler.Scheduler`, `sglang.srt.managers.scheduler.configure_scheduler`, `sglang.srt.observability.trace.process_tracing_init`, `sglang.srt.observability.trace.trace_set_thread_info`, `sglang.srt.utils.get_bool_env_var`, `sglang.srt.utils.kill_itself_when_parent_died`, `sglang.srt.utils.set_gpu_proc_affinity`, `sglang.srt.utils.numa_utils.get_numa_node_if_available`, `sglang.srt.utils.numa_utils.numa_bind_to_node`, `sglang.srt.environ.envs`, `sglang.utils.get_exception_traceback` |
| `areal/experimental/inference_service/sglang/rpc_proxy.py`     | `sglang.srt.managers.io_struct.RpcReqInput`, `sglang.srt.managers.io_struct.RpcReqOutput`, `sglang.srt.server_args.PortArgs`, `sglang.srt.utils.network.get_zmq_socket`                                                                                                                                                                                                                                                                                                                                                     |
| `areal/experimental/inference_service/sglang/bridge.py`        | HTTP-only (same protocol as `sglang_remote.py`). `/generate` request/response, `/pause_generation`, `/continue_generation`, `/release_memory_occupation`, `/resume_memory_occupation`.                                                                                                                                                                                                                                                                                                                                      |

### Secondary (model / infra layer)

| File                                                            | Imports / Usage                                                                                                                      |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `areal/infra/launcher/sglang_server.py`                         | HTTP-only. Polls `/v1/models` to confirm server readiness. Calls `SGLangConfig.build_cmd()` to construct the server launch command.  |
| `areal/api/cli_args.py`                                         | `SGLangConfig.build_cmd()`, `SGLangConfig.build_cmd_from_args()`, `SGLangConfig.build_args()`. Version check against `0.5.10.post1`. |
| `areal/trainer/rl_trainer.py`                                   | `SGLangConfig.build_args()` to construct `RemoteSGLangEngine`.                                                                       |
| `areal/infra/launcher/ray.py`                                   | `SGLangConfig` import + `to_structured_cfg`.                                                                                         |
| `areal/infra/launcher/local.py`                                 | `SGLangConfig` import + `to_structured_cfg`.                                                                                         |
| `areal/infra/launcher/slurm.py`                                 | `SGLangConfig` import + `to_structured_cfg`.                                                                                         |
| `areal/experimental/inference_service/controller/controller.py` | `SGLangConfig` in gateway controller.                                                                                                |
| `areal/experimental/inference_service/data_proxy/backend.py`    | HTTP protocol implementation for inference service proxy (SGLangBridgeBackend).                                                      |

### Tertiary (tests, config)

| File                                                 | Imports / Usage                                             |
| ---------------------------------------------------- | ----------------------------------------------------------- |
| `tests/grpo/test_grpo.py`                            | Test parameterization with `sglang` backend.                |
| `tests/experimental/openai/test_tool_call_parser.py` | `FunctionCallParser` / `ReasoningParser` integration tests. |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

> **Note:** Most sglang integration in AReaL is HTTP-based — the server is launched as a
> subprocess and communicated with over REST. Only `areal/experimental/openai/` imports
> the Python SDK directly. For the HTTP-based call sites, the risk is in **JSON field
> renames** in request payloads and response bodies, not in Python API signature
> changes.

______________________________________________________________________

### 1. `POST /generate` — request payload

**Source:** `python/sglang/srt/entrypoints/http_server.py` (request handler)

Called in `areal/engine/sglang_remote.py` (`SGLangBackend.build_generation_request`,
lines 42–88):

```python
sample_params = {
    "top_p": gconfig.top_p,
    "top_k": gconfig.top_k,
    "max_new_tokens": gconfig.max_new_tokens,
    "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
    "stop_token_ids": stop_token_ids,
    "ignore_eos": gconfig.ignore_eos,
    "skip_special_tokens": gconfig.skip_special_tokens,
    "frequency_penalty": gconfig.frequency_penalty,
}
if stop:
    sample_params["stop"] = stop

payload = {
    "input_ids": req.input_ids.copy(),
    "image_data": req.image_data,
    "sampling_params": sample_params,
    "return_logprob": True,
    "stream": False,
}
# conditionally added: "return_routed_experts": True, "lora_path": <str>
```

**Check:** Confirm `input_ids`, `image_data`, `sampling_params`, `return_logprob`,
`lora_path`, and `return_routed_experts` are still accepted top-level fields. Verify all
`sampling_params` sub-keys are unchanged (`top_p`, `top_k`, `max_new_tokens`,
`temperature`, `stop_token_ids`, `ignore_eos`, `skip_special_tokens`,
`frequency_penalty`, `stop`). Check whether `stream=False` is still the correct way to
request a non-streaming response.

______________________________________________________________________

### 2. `POST /generate` — response parsing

**Source:** `python/sglang/srt/entrypoints/http_server.py` (response schema)

Parsed in `areal/engine/sglang_remote.py` (`SGLangBackend.parse_generation_response`,
lines 90–126):

```python
meta_info = response["meta_info"]
finish_reason = meta_info["finish_reason"]
stop_reason = finish_reason["type"]
stop_message = finish_reason.get("message", "")
routed_experts = meta_info.get("routed_experts", None)
if routed_experts is not None:
    num_sgl_token = meta_info["prompt_tokens"] + meta_info["completion_tokens"] - 1
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts.encode("utf-8")), dtype=np.int32
    ).reshape(num_sgl_token, -1)
output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]
```

**Check:** Confirm `meta_info` is still a top-level key. Verify `finish_reason` is still
a dict with a `"type"` key (not a plain string). Verify `output_token_logprobs` is still
a list of `[logprob, token_id]` pairs. Check that `prompt_tokens` and
`completion_tokens` are still present when `return_logprob=True`. Confirm
`routed_experts` is still base64-encoded int32 that can be decoded via
`pybase64.b64decode` and reshaped to `(num_tokens, num_layers*expert_top_k)`.

______________________________________________________________________

### 3. `POST /load_lora_adapter` — request payload

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/engine/sglang_remote.py`
(`SGLangBackend.build_disk_weight_update_requests`, lines 139–144):

```python
HttpRequest(
    endpoint="/load_lora_adapter",
    payload={"lora_name": lora_name, "lora_path": str(meta.path)},
)
```

**Check:** Confirm the endpoint still exists and accepts `lora_name` and `lora_path`.
Check whether the endpoint was renamed or merged into another route.

______________________________________________________________________

### 4. `POST /update_weights_from_disk` — request payload

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/engine/sglang_remote.py`
(`SGLangBackend.build_disk_weight_update_requests`, lines 148–158):

```python
HttpRequest(
    endpoint="/update_weights_from_disk",
    payload={
        "model_path": str(meta.path),
        "abort_all_requests": True,
    },
)
```

**Check:** Confirm `model_path` and `abort_all_requests` are still valid payload keys.
Check whether the endpoint was renamed (e.g., to `/update_weights`).

______________________________________________________________________

### 5. `POST /update_weights_from_distributed` — request payload

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/engine/sglang_remote.py`
(`SGLangBackend.build_distributed_weight_update_requests`, lines 173–186):

```python
HttpRequest(
    endpoint="/update_weights_from_distributed",
    payload={
        "names": [pspec.name for pspec in param_specs],
        "dtypes": [pspec.dtype for pspec in param_specs],
        "shapes": [pspec.shape for pspec in param_specs],
        "group_name": meta.nccl_group_name,
        "abort_all_requests": True,
    },
)
```

**Check:** Confirm all five payload keys are unchanged. Verify `dtypes` still accepts
string representations of torch dtypes. Check whether `abort_all_requests` is still
honored.

______________________________________________________________________

### 6. `POST /init_weights_update_group` — request payload

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/engine/sglang_remote.py`
(`SGLangBackend.build_init_weights_group_request`, lines 188–207):

```python
HttpRequest(
    endpoint="/init_weights_update_group",
    payload={
        "master_address": format_host_for_url(meta.nccl_master_address),
        "master_port": str(meta.nccl_master_port),
        "rank_offset": rank_offset,
        "world_size": gen_parallel.world_size + 1,
        "backend": current_platform.communication_backend,
        "group_name": meta.nccl_group_name,
    },
)
```

**Check:** Confirm all six payload keys are still accepted. Verify `rank_offset`
semantics (whether the training-rank offset convention changed). Check `backend`
accepted values (e.g., `"nccl"`, `"gloo"`).

______________________________________________________________________

### 7. `POST /pause_generation` and `POST /continue_generation`

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/engine/sglang_remote.py` (`get_pause_request` line 211,
`get_resume_request` line 215):

```python
HttpRequest(endpoint="/pause_generation", payload={})
HttpRequest(endpoint="/continue_generation", payload={})
```

**Check:** Confirm both endpoints still exist with these exact names. Check whether they
now require any payload fields (e.g., `wait_for_inflight_requests`).

______________________________________________________________________

### 8. `GET /health`

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/engine/sglang_remote.py` (`get_health_check_request`, line 219):

```python
HttpRequest(endpoint="/health", payload={}, method="GET")
```

**Check:** Confirm the endpoint is still a `GET` request and returns a 2xx on success.
Verify it doesn't require authentication headers.

______________________________________________________________________

### 9. `GET /v1/models` — server readiness poll

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/infra/launcher/sglang_server.py` (`wait_for_server`, lines 65–84):

```python
response = requests.get(
    f"{base_url}/v1/models",
    headers={"Authorization": "Bearer None"},
)
if response.status_code == 200:
    break
```

**Check:** Confirm `/v1/models` still returns 200 once the server is ready. Check
whether the `Authorization: Bearer None` header is still accepted (or if auth
requirements changed). If SGLang changes its readiness probe endpoint, update this
polling logic.

______________________________________________________________________

### 10. `POST /release_memory_occupation` and `POST /resume_memory_occupation`

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/engine/sglang_remote.py` (`get_offload_request` line 223,
`get_onload_request` lines 225–234):

```python
# offload (release GPU memory)
HttpRequest(endpoint="/release_memory_occupation", payload={})

# onload (resume GPU memory, optional tags for multi-stage resume)
payload = {"tags": tags} if tags is not None else {}
HttpRequest(endpoint="/resume_memory_occupation", payload=payload)
```

**Check:** Confirm both endpoints still exist. For `/resume_memory_occupation`, verify
`tags` is still an optional list of strings (available tags: `"weights"`, `"kv_cache"`).
Check if additional fields are now required for either endpoint.

______________________________________________________________________

### 11. `sglang.srt.entrypoints.openai.protocol.Function` and `.Tool`

**Source:** `python/sglang/srt/entrypoints/openai/protocol.py`

Called in `areal/experimental/openai/tool_call_parser.py` (`_process_tool_calls_sglang`,
lines 73–94):

```python
from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglTool

SglTool(
    type=tool["type"],
    function=SglFunction(
        name=tool.get("name"),
        description=tool.get("description"),
        parameters=tool.get("parameters"),
    ),
)
# or:
SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
```

**Check:** Confirm `Function` and `Tool` are still exported from this module path (not
moved to `sglang.srt.protocol` or similar). Verify `Function` still accepts `name`,
`description`, `parameters`. Verify `Tool` still accepts `type` and `function`. Check
for newly required fields.

______________________________________________________________________

### 12. `sglang.srt.function_call.function_call_parser.FunctionCallParser`

**Source:** `python/sglang/srt/function_call/function_call_parser.py`

Called in `areal/experimental/openai/tool_call_parser.py` (`_process_tool_calls_sglang`,
lines 96, 105, 109):

```python
from sglang.srt.function_call.function_call_parser import FunctionCallParser

parser_p = FunctionCallParser(tools, tool_call_parser)
# Public methods called:
parser_p.has_tool_call(content_text)       # -> bool
content_text, call_info_list = parser_p.parse_non_stream(content_text)
# call_info_list items expose: call_info.name, call_info.parameters
```

**Check:** Confirm the module path is unchanged. Verify the constructor still takes
`(tools, tool_call_parser)` positionally where `tools` is a list of `SglTool` objects.
Confirm `has_tool_call(text) -> bool` and `parse_non_stream(text) -> (str, list)`
methods still exist with the same signatures. Verify returned `call_info` objects still
expose `.name` and `.parameters` attributes.

______________________________________________________________________

### 13. `sglang.srt.parser.reasoning_parser.ReasoningParser`

**Source:** `python/sglang/srt/parser/reasoning_parser.py`

Called in `areal/experimental/openai/tool_call_parser.py` (`_process_tool_calls_sglang`,
lines 76, 97, 101–102):

```python
from sglang.srt.parser.reasoning_parser import ReasoningParser

reasoning_parser_p = ReasoningParser(reasoning_parser)
# Accessed attributes:
reasoning_parser_p.detector.think_start_token
reasoning_parser_p.detector.think_end_token
```

**Check:** Confirm the module path is unchanged (not moved to
`sglang.srt.reasoning_parser`). Verify the constructor still accepts a single parser
name string. Critically, verify the returned instance still exposes
`.detector.think_start_token` and `.detector.think_end_token` — these are used to
extract `<think>` blocks from model output.

______________________________________________________________________

### 14. `SGLangConfig.build_cmd` / CLI flag compatibility

**Source:** `python/sglang/launch_server.py`, `python/sglang/srt/server_args.py`

Called in `areal/api/cli_args.py` via `SGLangConfig.build_args()` and
`SGLangConfig.build_cmd()`:

```python
# build_args() assembles a dict of CLI flags from SGLangConfig fields
# build_cmd() converts those flags into a subprocess command list
args = SGLangConfig.build_args()
# flags like --model-path, --tp-size, --max-loaded-loras, etc.
```

**Check:** For each of the ~70 config fields in `SGLangConfig`, confirm the
corresponding `sglang` CLI flag still exists with the same name and accepted values. Pay
special attention to any fields that were added in recent releases (e.g.,
`max_loaded_loras` added in 0.4.10.post2). Check `python/sglang/srt/server_args.py` for
added/removed/renamed `argparse` arguments.

______________________________________________________________________

### 15. `Engine._launch_subprocesses` — server lifecycle

**Source:** `python/sglang/srt/entrypoints/engine.py`

Called in `areal/experimental/inference_service/sglang/launch_server.py`
(`areal_launch_server`):

```python
from sglang.srt.entrypoints.engine import Engine, init_tokenizer_manager

(
    tokenizer_manager,
    template_manager,
    port_args,
    scheduler_init_result,
    subprocess_watchdog,
) = Engine._launch_subprocesses(
    server_args=server_args,
    init_tokenizer_manager_func=init_tokenizer_manager,
    run_scheduler_process_func=areal_run_scheduler_process,
    run_detokenizer_process_func=run_detokenizer_process,
)
```

**Check:** Confirm `_launch_subprocesses` is still a classmethod. Verify it still
accepts `run_scheduler_process_func` as a callable parameter for scheduler process
injection. Verify the return tuple is
`(TokenizerManager, TemplateManager, PortArgs, SchedulerInitResult, SubprocessWatchdog)`.
Confirm `SchedulerInitResult.scheduler_infos` attribute exists.

______________________________________________________________________

### 16. `_setup_and_run_http_server` — HTTP server bootstrap

**Source:** `python/sglang/srt/entrypoints/http_server.py`

Called in `areal/experimental/inference_service/sglang/launch_server.py`
(`areal_launch_server`):

```python
from sglang.srt.entrypoints.http_server import (
    _execute_server_warmup,
    _setup_and_run_http_server,
    app,
)

_setup_and_run_http_server(
    server_args,
    tokenizer_manager,
    template_manager,
    port_args,
    scheduler_init_result.scheduler_infos,
    subprocess_watchdog,
    execute_warmup_func=_execute_server_warmup,
)
```

**Check:** Confirm the function still exists at this module path. Verify signature:
`(server_args, tokenizer_manager, template_manager, port_args, scheduler_infos, subprocess_watchdog, execute_warmup_func, launch_callback)`.
Confirm `app` is still the FastAPI instance exported from this module.

______________________________________________________________________

### 17. `Scheduler` constructor and `configure_scheduler`

**Source:** `python/sglang/srt/managers/scheduler.py`

Called in `areal/experimental/inference_service/sglang/scheduler.py`
(`areal_run_scheduler_process`):

```python
from sglang.srt.managers.scheduler import Scheduler, configure_scheduler

dp_rank = configure_scheduler(
    server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank
)

scheduler = Scheduler(
    server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank,
    attn_cp_rank, moe_dp_rank, dp_rank,
)
pipe_writer.send(scheduler.get_init_info())
scheduler.run_event_loop()
```

**Check:** Confirm `configure_scheduler` signature is
`(server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank)`.
Confirm `Scheduler.__init__` signature is
`(server_args, port_args, gpu_id, tp_rank, moe_ep_rank, pp_rank, attn_cp_rank, moe_dp_rank, dp_rank)`.
Verify `get_init_info()` and `run_event_loop()` still exist. Confirm `tp_rank`,
`dp_rank`, `tp_cpu_group`, `tp_size` attributes are present on the scheduler instance
(used by `AwexSchedulerBridge`).

______________________________________________________________________

### 18. `RpcReqInput` / `RpcReqOutput` — scheduler RPC protocol

**Source:** `python/sglang/srt/managers/io_struct.py`

Used in `areal/experimental/inference_service/sglang/rpc_proxy.py`:

```python
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput

req = RpcReqInput(method=method, parameters=kwargs if kwargs else None)
self._rpc_socket.send_pyobj(req)
resp: RpcReqOutput = self._rpc_socket.recv_pyobj()
# resp.success: bool, resp.message: str
```

**Check:** Confirm `RpcReqInput` still has `method: str` and
`parameters: Optional[Dict]` fields. Confirm `RpcReqOutput` still has `success: bool`
and `message: str` fields. Verify `send_pyobj`/`recv_pyobj` serialization still works
(both inherit from `BaseReq` which is a dataclass).

______________________________________________________________________

### 19. `sglang.srt.utils` module reorganization (v0.5.10+)

**Source:** `python/sglang/srt/utils/__init__.py`, `python/sglang/srt/utils/common.py`,
`python/sglang/srt/utils/network.py`, `python/sglang/srt/utils/numa_utils.py`

In v0.5.10+, `sglang.srt.utils/__init__.py` only re-exports from `common.py` via
`from sglang.srt.utils.common import *`. Functions in submodules must be imported
directly:

- `get_zmq_socket` → `sglang.srt.utils.network`
- `get_numa_node_if_available`, `numa_bind_to_node` → `sglang.srt.utils.numa_utils`
- `kill_itself_when_parent_died`, `get_bool_env_var`, `set_gpu_proc_affinity`,
  `kill_process_tree` → `sglang.srt.utils` (via `common.py`)

**Check:** On each upgrade, verify `sglang.srt.utils/__init__.py` re-export scope. If
functions move between submodules, update AReaL import paths accordingly.

______________________________________________________________________

## Version-Guarded Code

- `areal/api/cli_args.py:1877-1878` —
  `if not pkg_version.is_version_greater_or_equal("sglang", "0.5.10.post1"): raise RuntimeError(...)`.
  Minimum floor raised from `0.4.9.post2` to `0.5.10.post1` during this upgrade. The
  legacy `max_loaded_loras` version guard (`<0.4.10.post2`) was removed because the new
  floor is well past that version.
