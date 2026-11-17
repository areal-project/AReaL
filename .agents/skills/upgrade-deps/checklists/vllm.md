---
package: vllm
github: vllm-project/vllm
branch_template: v${VERSION}
upstream_paths:
  - vllm/entrypoints/openai/
  - vllm/entrypoints/launchers/
  - vllm/entrypoints/serve/
  - vllm/v1/engine/async_llm.py
  - vllm/lora/
  - vllm/model_executor/model_loader/
  - vllm/v1/worker/gpu_worker.py
  - vllm/worker/worker.py
  - vllm/reasoning/
  - vllm/tool_parsers/
  - vllm/distributed/parallel_state.py
  - vllm/utils/argparse_utils.py
---

## Affected Files

### Primary (engine layer — most likely to break)

| File                                             | Imports / Usage                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/vllm_ext/vllm_worker_extension.py` | `vllm.logger.init_logger`, `vllm.lora.lora_model.LoRAModel`, `vllm.lora.peft_helper.PEFTHelper`, `vllm.lora.request.LoRARequest`, `vllm.model_executor.model_loader.get_model_loader`, `vllm.model_executor.model_loader.reload.initialize_layerwise_reload` / `.finalize_layerwise_reload`, `vllm.config.set_current_vllm_config`, `model_runner.reload_weights`; **private APIs**: `_adapter_manager._registered_adapters`, `._add_adapter`, `.activate_adapter`                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `areal/engine/vllm_ext/areal_vllm_server.py`     | `vllm.entrypoints.openai.api_server.build_app`, `.run_server`; `vllm.entrypoints.common.cli_args.make_arg_parser`, `.validate_parsed_serve_args` (≥0.33; fallbacks `launchers.cli_args` for 0.32, `openai.cli_args` before); `vllm.entrypoints.openai.completion.api_router.create_completion`; `vllm.entrypoints.openai.completion.protocol.CompletionRequest`; `vllm.entrypoints.serve.engine.protocol.ErrorResponse`, `.OpenAIBaseModel` (≥0.32; fallback `openai.engine.protocol`); `vllm.entrypoints.serve.utils.api_utils.load_aware_call`, `.validate_json_request`, `.with_cancellation` (≥0.28; fallback `openai.utils` / `entrypoints.utils`); `vllm.entrypoints.common.startup.cli_env_setup` (≥0.33; fallbacks `serve.utils.api_utils`, `entrypoints.utils`); `engine_client.pause_generation(mode=...)`; `vllm.lora.request.LoRARequest`; `vllm.utils.argparse_utils.FlexibleArgumentParser` |
| `areal/engine/vllm_remote.py`                    | HTTP-only — builds vLLM launch commands via `vLLMConfig.build_cmd_from_args()`; sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `areal/utils/vllm_response.py`                   | HTTP-only — parses `/v1/completions` and chat `logprobs` JSON (`tokens`/`token_logprobs` or `content[].token`/`.logprob`); rejects vLLM's `-9999.0` missing/clamped-logprob sentinel. Used by `vllm_remote.py` and `areal/v2/inference_service/vllm/bridge.py`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

### Secondary (model / infra layer)

| File                                            | Imports / Usage                                                                              |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `areal/infra/platforms/cuda.py`                 | `vllm.v1.worker.gpu_worker.Worker` (V1) with fallback to `vllm.worker.worker.Worker` (V0)    |
| `areal/infra/platforms/unknown.py`              | same V1/V0 Worker import pattern                                                             |
| `areal/experimental/openai/tool_call_parser.py` | `vllm.reasoning.ReasoningParserManager`, `vllm.tool_parsers.ToolParserManager` (conditional) |

### Tertiary (tests, config)

| File                                                 | Imports / Usage                          |
| ---------------------------------------------------- | ---------------------------------------- |
| `tests/experimental/openai/test_tool_call_parser.py` | unit tests with vLLM mocking             |
| `tests/test_inference_engines.py`                    | integration tests for `RemotevLLMEngine` |

### External (awex — separate subsystem)

| File                                          | Imports / Usage                                                                                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `asystem-awex/awex/vllm_plugin.py`            | `vllm.entrypoints.openai.api_server.router`, `vllm.distributed.parallel_state.*`, `vllm.v1.worker.worker_base.WorkerBase`, `vllm.model_executor.model_loader.get_model_loader` |
| `asystem-awex/awex/awex_vllm_server.py`       | `vllm.entrypoints.openai.api_server.run_server`, cli_args, `FlexibleArgumentParser`                                                                                            |
| `asystem-awex/awex/engine/vllm.py`            | VLLMEngine wrapper                                                                                                                                                             |
| `asystem-awex/awex/vllm_awex_adapter.py`      | `AwexVLLMServerAdapter` for runtime weight updates                                                                                                                             |
| `asystem-awex/awex/sharding/vllm_sharding.py` | sharding strategy helpers                                                                                                                                                      |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. `vllm.entrypoints.openai.api_server.build_app` and `.run_server`

**Source:** `vllm/entrypoints/openai/api_server.py` (re-export);
`vllm/entrypoints/launchers/api_server/entry.py` (definition, ≥0.32)

Called in `areal/engine/vllm_ext/areal_vllm_server.py`:

```python
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
from vllm.entrypoints.openai.api_server import run_server

# build_app is monkey-patched in the module where run_server looks it up:
# vllm.entrypoints.openai.api_server before 0.32,
# vllm.entrypoints.launchers.api_server.entry from 0.32 on.
_api_server_module = sys.modules[run_server.__module__]
_api_server_module.build_app = _areal_build_app  # replaces the module-level reference

# _areal_build_app calls the original with supported_tasks kwarg (line 371):
app = _original_build_app(args, supported_tasks=supported_tasks)

# run_server is called via uvloop (line 398):
uvloop.run(run_server(args))
```

**Check:** Confirm both functions still exist at this path. Verify `build_app` accepts
`(args, supported_tasks=..., model_config=...)` and returns a FastAPI `app`. Verify
`run_server` is an async function accepting `(args)`. Check that monkey-patching
`build_app` in `sys.modules[run_server.__module__]` still works (i.e., `run_server`
reads `build_app` from its own module, not a closure).

______________________________________________________________________

### 2. `vllm.entrypoints.common.cli_args.make_arg_parser` and `.validate_parsed_serve_args`

**Source:** `vllm/entrypoints/common/cli_args.py` (≥0.33);
`vllm/entrypoints/launchers/cli_args.py` (0.32); `vllm/entrypoints/openai/cli_args.py`
(\<0.32)

Called in `areal/engine/vllm_ext/areal_vllm_server.py`:

```python
try:
    from vllm.entrypoints.common.cli_args import make_arg_parser, validate_parsed_serve_args  # >=0.33
except ImportError:
    try:
        from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args  # 0.32
    except ImportError:
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
parser = make_arg_parser(parser)
args = parser.parse_args()
validate_parsed_serve_args(args)
```

**Check:** Verify `make_arg_parser` still accepts a parser object and returns an
augmented parser. Check `validate_parsed_serve_args` for new validation rules that might
reject AReaL's custom args.

______________________________________________________________________

### 3. `vllm.entrypoints.openai.completion` — protocol and router

**Source:** `vllm/entrypoints/openai/completion/`

Called in `areal/engine/vllm_ext/areal_vllm_server.py`:

```python
from vllm.entrypoints.openai.completion.api_router import create_completion as original_create_completion
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
# >=0.32: vllm.entrypoints.serve.engine.protocol; <0.32: vllm.entrypoints.openai.engine.protocol
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, OpenAIBaseModel
# >=0.28: vllm.entrypoints.serve.utils.api_utils; older: vllm.entrypoints.openai.utils
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
```

**Check:** Verify the completion sub-package still exists (was restructured in some vLLM
versions). Confirm `CompletionRequest` Pydantic model fields. Verify `ErrorResponse` and
`OpenAIBaseModel` are still at `serve.engine.protocol`. Check `validate_json_request`
still returns a FastAPI `Depends`.

______________________________________________________________________

### 4. `vllm.entrypoints.serve.utils.api_utils` / `common.startup` — server utilities

**Source:** `vllm/entrypoints/serve/utils/api_utils.py` (≥0.28);
`vllm/entrypoints/utils.py` (older); `cli_env_setup` is in
`vllm/entrypoints/common/startup.py` from 0.33

Called in `areal/engine/vllm_ext/areal_vllm_server.py`:

```python
try:
    from vllm.entrypoints.serve.utils.api_utils import (
        load_aware_call, validate_json_request, with_cancellation,
    )
except ImportError:
    from vllm.entrypoints.openai.utils import validate_json_request
    from vllm.entrypoints.utils import load_aware_call, with_cancellation

try:
    from vllm.entrypoints.common.startup import cli_env_setup  # >=0.33
except ImportError:
    try:
        from vllm.entrypoints.serve.utils.api_utils import cli_env_setup  # 0.28 - 0.32
    except ImportError:
        from vllm.entrypoints.utils import cli_env_setup

# Used as stacked decorators on the create_completion endpoint (lines 349-350):
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    ...

# Called at server startup (line 390):
cli_env_setup()
```

**Check:** Confirm the three decorators are still exported from `serve.utils.api_utils`
and `cli_env_setup` from `common.startup`. Verify `with_cancellation` is still a
decorator compatible with async endpoints. Verify `load_aware_call` is still a decorator
(not renamed to `load_aware`). Confirm `cli_env_setup()` is still a no-arg function.

______________________________________________________________________

### 5. `vllm.lora.request.LoRARequest`

**Source:** `vllm/lora/request.py`

Called in `areal/engine/vllm_ext/vllm_worker_extension.py` (lines 10, 61-66) and
`areal_vllm_server.py` (line 21):

```python
from vllm.lora.request import LoRARequest

lora_request = LoRARequest(
    lora_name=lora_name,
    lora_int_id=lora_int_id,
    lora_path=lora_model_path,
    base_model_name=base_model_name,
)
```

**Check:** Verify constructor kwargs unchanged. Confirm `base_model_name` is still
accepted (vs `base_model_name_or_path`). Check for new required fields.

______________________________________________________________________

### 6. `vllm.lora.lora_model.LoRAModel.from_lora_tensors`

**Source:** `vllm/lora/lora_model.py`

Called in `areal/engine/vllm_ext/vllm_worker_extension.py`:

```python
from vllm.lora.lora_model import LoRAModel

lora_model = LoRAModel.from_lora_tensors(
    lora_model_id=int_id,
    tensors=tensors,
    peft_helper=peft_helper,
    device=device,
    dtype=dtype,
    model_vocab_size=vocab_size,
    weights_mapper=weights_mapper,
)
```

**Check:** Verify all keyword arguments. `from_lora_tensors` is a classmethod — confirm
it still exists and returns a `LoRAModel`. Check `peft_helper` type (should be
`PEFTHelper`). Verify `weights_mapper` is still accepted.

______________________________________________________________________

### 7. `vllm.lora.peft_helper.PEFTHelper.from_dict`

**Source:** `vllm/lora/peft_helper.py`

Called in `areal/engine/vllm_ext/vllm_worker_extension.py`:

```python
from vllm.lora.peft_helper import PEFTHelper

peft_helper = PEFTHelper.from_dict(peft_config_dict)
```

**Check:** Verify `from_dict` classmethod still exists. Confirm it accepts a dict with
keys `r`, `lora_alpha`, `target_modules`, `bias`. Check return type.

______________________________________________________________________

### 8. `vllm.model_executor.model_loader.get_model_loader`

**Source:** `vllm/model_executor/model_loader/`

Called in `areal/engine/vllm_ext/vllm_worker_extension.py` (line 34):

```python
from vllm.model_executor.model_loader import get_model_loader

model_loader = get_model_loader(self.model_runner.vllm_config.load_config)
model_loader.load_weights(self.model_runner.model, model_config=self.model_runner.model_config)
```

**Check:** Verify function exists. Confirm it accepts `vllm_config.load_config` (type
`LoadConfig`). Verify the returned loader has a `load_weights(model, model_config=...)`
method (not `load_model`).

______________________________________________________________________

### 9. LoRA adapter manager private APIs (HIGH RISK)

**Source:** `vllm/lora/` (internal, unstable)

Called in `areal/engine/vllm_ext/vllm_worker_extension.py` (lines 154-293):

```python
# Diagnostic read — public list_adapters() at line 179:
adapter_ids = self.model_runner.lora_manager.list_adapters()

# Private read — _registered_adapters at line 187:
lora_model = self.model_runner.lora_manager._adapter_manager._registered_adapters[lora_int_id]

# Public remove — remove_adapter() at line 270:
self.model_runner.lora_manager.remove_adapter(lora_int_id)

# Private write — _add_adapter / activate_adapter at lines 272-274:
self.model_runner.lora_manager._adapter_manager._add_adapter(new_lora_model)
self.model_runner.lora_manager._adapter_manager.activate_adapter(new_lora_model.id)
```

The code comments at line 155-156 explicitly note: _"This code relies on vLLM private
APIs: `_adapter_manager`, `_registered_adapters`, and `_add_adapter`/`activate_adapter`,
which may change/breakdown due to newer vllm versions."_

**Check:** These are **private APIs** — highest breakage risk. Verify `lora_manager`
still exposes `list_adapters()` (public), `remove_adapter()` (public), and
`_adapter_manager` (private). Confirm `_registered_adapters` is still a dict-like
mapping of `int_id -> LoRAModel`. Confirm `_add_adapter` and `activate_adapter` method
signatures. Check if public alternatives now exist that replace the private path.

______________________________________________________________________

### 10. `vllm.v1.worker.gpu_worker.Worker` / `vllm.worker.worker.Worker`

**Source:** `vllm/v1/worker/gpu_worker.py`, `vllm/worker/worker.py`

Called in `areal/infra/platforms/cuda.py` and `unknown.py`:

```python
try:
    from vllm.v1.worker.gpu_worker import Worker  # V1
except ImportError:
    from vllm.worker.worker import Worker  # V0 fallback
```

**Check:** Confirm at least one of these import paths works. If V0 is being removed,
update to V1 only. Check `Worker` class for method compatibility.

______________________________________________________________________

### 11. `vllm.reasoning.ReasoningParserManager` and `vllm.tool_parsers.ToolParserManager`

**Source:** `vllm/reasoning/`, `vllm/tool_parsers/`

Called in `areal/experimental/openai/tool_call_parser.py` (lines 157-206, inside
`_process_tool_calls_vllm`):

```python
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

# get_reasoning_parser returns a CLASS, then instantiated with tokenizer (line 164-167):
reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
reasoning_parser_inst = reasoning_parser_cls(tokenizer)
# Accessed attributes: reasoning_parser_inst.start_token, reasoning_parser_inst.end_token

# get_tool_parser also returns a CLASS, then instantiated with tokenizer (line 191-206):
tool_parser_cls = ToolParserManager.get_tool_parser(vllm_name)
tool_parser = tool_parser_cls(tokenizer)
# Called methods: tool_parser.extract_tool_calls(model_output, request)
```

**Check:** Confirm `get_reasoning_parser` and `get_tool_parser` return **classes** (not
instances). Verify the classes accept `(tokenizer)` in their constructor. Confirm
`start_token`/`end_token` attributes on reasoning parser instances. Verify
`extract_tool_calls` method signature on tool parser instances. These are used as
fallback when SGLang parsers are unavailable.

______________________________________________________________________

### 12. `vllm.utils.argparse_utils.FlexibleArgumentParser`

**Source:** `vllm/utils/argparse_utils.py`

Called in `areal/engine/vllm_ext/areal_vllm_server.py`:

```python
from vllm.utils.argparse_utils import FlexibleArgumentParser
```

**Check:** Verify this class still exists at this path (it was added in a specific vLLM
version). `asystem-awex/awex/awex_vllm_server.py` already has a try/except fallback for
this import.

______________________________________________________________________

### 13. `vllm.logger.init_logger`

**Source:** `vllm/logger.py`

Called in `areal/engine/vllm_ext/vllm_worker_extension.py` and `areal_vllm_server.py`:

```python
from vllm.logger import init_logger
logger = init_logger(__name__)
```

**Check:** Verify function exists. Confirm it returns a standard
`logging.Logger`-compatible object.

______________________________________________________________________

### 14. vLLM CLI flag compatibility

**Source:** `vllm/engine/arg_utils.py`, `vllm/entrypoints/openai/cli_args.py`

Called indirectly via `vLLMConfig.build_cmd_from_args()` in `areal/api/cli_args.py`:

```python
# Assembles CLI flags: --model, --tensor-parallel-size, --gpu-memory-utilization,
# --max-model-len, --enable-lora, --max-lora-rank, etc.
```

**Check:** For each field in `vLLMConfig`, confirm the corresponding vLLM CLI flag still
exists. Pay attention to `--enable-lora` and related LoRA flags. Verify env var
`VLLM_ALLOW_RUNTIME_LORA_UPDATING` is still respected.

vLLM 0.33 renamed the `--no-enable-prefix-caching` / `--no-enable-chunked-prefill` flags
to `--disable-*` (`EnableDisableAction`) but still accepts the old spelling with the
same result; AReaL's `vLLMConfig.no_enable_*` fields still emit the old spelling. If a
future vLLM drops it, the command will be rejected at startup.

A quick check: build the parser with `make_arg_parser(FlexibleArgumentParser())`, then
confirm every `dataclasses.fields(vLLMConfig)` name maps to a known `--flag` and that
`parser.parse_args(vLLMConfig.build_cmd_from_args(...)[3:])` accepts the default
command.

______________________________________________________________________

### 15. `AsyncLLM.pause_generation` (engine client)

**Source:** `vllm/v1/engine/async_llm.py`

Called in `areal/engine/vllm_ext/areal_vllm_server.py` before every weight update and in
`areal_pause_generation`:

```python
llm = raw_request.app.state.engine_client
await llm.pause_generation(mode="abort", clear_cache=True)
await llm.reset_mm_cache()
```

**Check:** Verify `mode` still accepts `"abort"` and `clear_cache` is still a keyword.
`wait_for_inflight_requests` was deprecated in 0.19 and removed in 0.32; `mode` exists
in 0.19.1+, so do not reintroduce the old keyword.

______________________________________________________________________

### 16. Logprob sentinel in OpenAI responses

**Source:** `vllm/entrypoints/openai/chat_completion/protocol.py`
(`ChatCompletionLogProb.logprob` default), `.../chat_completion/serving.py`,
`.../completion/serving.py` (`max(logprob, -9999.0)` clamp)

Relied on by `areal/utils/vllm_response.py` (`_AMBIGUOUS_VLLM_LOGPROB = -9999.0`):

**Check:** Confirm vLLM still uses `-9999.0` as both the default and the clamp floor. If
it changes, update `_AMBIGUOUS_VLLM_LOGPROB` and the upstream source links in that file.

______________________________________________________________________

## Version-Guarded Code

- `areal/engine/vllm_ext/areal_vllm_server.py` — `common.cli_args` / `common.startup`
  (≥0.33), `launchers.cli_args` / `serve.engine.protocol` (≥0.32) and
  `serve.utils.api_utils` (≥0.28) imports fall back to the pre-0.32 `openai.*` /
  `entrypoints.utils` paths. Keep while `Dockerfile.rocm` still builds vLLM 0.19.1;
  remove once every image is on ≥0.32.

- `areal/engine/vllm_ext/vllm_worker_extension.py` — `set_current_vllm_config` falls
  back to a no-op context manager for vLLM \< 0.19. Every supported image is ≥0.19.1, so
  the fallback can be removed.

- `areal/infra/platforms/cuda.py` and `unknown.py` — tries
  `vllm.v1.worker.gpu_worker.Worker` first, falls back to `vllm.worker.worker.Worker`.
  Once V0 support is dropped, remove the fallback.

- `asystem-awex/awex/awex_vllm_server.py` — `FlexibleArgumentParser` wrapped in
  try/except with fallback to `argparse.ArgumentParser`.

- `asystem-awex/awex/vllm_plugin.py` — `get_pcp_group` import wrapped in try/except
  (added in newer vLLM versions).
