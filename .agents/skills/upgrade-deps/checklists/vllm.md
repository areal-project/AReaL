---
package: vllm
github: vllm-project/vllm
branch_template: v${VERSION}
upstream_paths:
  - vllm/entrypoints/openai/
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

| File                                             | Imports / Usage                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/vllm_ext/vllm_worker_extension.py` | `vllm.logger.init_logger`, `vllm.lora.lora_model.LoRAModel`, `vllm.lora.peft_helper.PEFTHelper`, `vllm.lora.request.LoRARequest`, `vllm.model_executor.model_loader.get_model_loader`; **private APIs**: `_adapter_manager._registered_adapters`, `._add_adapter`, `.activate_adapter`                                                                                                                                                                                                                                                                                                          |
| `areal/engine/vllm_ext/areal_vllm_server.py`     | `vllm.entrypoints.openai.api_server.build_app`, `.run_server`; `vllm.entrypoints.openai.cli_args.make_arg_parser`, `.validate_parsed_serve_args`; `vllm.entrypoints.openai.completion.api_router.create_completion`; `vllm.entrypoints.openai.completion.protocol.CompletionRequest`; `vllm.entrypoints.openai.engine.protocol.ErrorResponse`, `.OpenAIBaseModel`; `vllm.entrypoints.openai.utils.validate_json_request`; `vllm.entrypoints.utils.cli_env_setup`, `.load_aware_call`, `.with_cancellation`; `vllm.lora.request.LoRARequest`; `vllm.utils.argparse_utils.FlexibleArgumentParser` |
| `areal/engine/vllm_remote.py`                    | HTTP-only — builds vLLM launch commands via `vLLMConfig.build_cmd_from_args()`; sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

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

**Source:** `vllm/entrypoints/openai/api_server.py`

Called in `areal/engine/vllm_ext/areal_vllm_server.py` (lines 10-11, 366-398):

```python
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
from vllm.entrypoints.openai.api_server import run_server

# build_app is monkey-patched at line 388 to inject AReaL routes:
import vllm.entrypoints.openai.api_server as _api_server_module
_api_server_module.build_app = _areal_build_app  # replaces the module-level reference

# _areal_build_app calls the original with supported_tasks kwarg (line 371):
app = _original_build_app(args, supported_tasks=supported_tasks)

# run_server is called via uvloop (line 398):
uvloop.run(run_server(args))
```

**Check:** Confirm both functions still exist at this path. Verify `build_app` accepts
`(args, supported_tasks=...)` and returns a FastAPI `app`. Verify `run_server` is an
async function accepting `(args)`. Check that monkey-patching
`_api_server_module.build_app` still works (i.e., `run_server` reads `build_app` from
the module, not a closure).

______________________________________________________________________

### 2. `vllm.entrypoints.openai.cli_args.make_arg_parser` and `.validate_parsed_serve_args`

**Source:** `vllm/entrypoints/openai/cli_args.py`

Called in `areal/engine/vllm_ext/areal_vllm_server.py` (lines 12, 391-396):

```python
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
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, OpenAIBaseModel
from vllm.entrypoints.openai.utils import validate_json_request
```

**Check:** Verify the completion sub-package still exists (was restructured in some vLLM
versions). Confirm `CompletionRequest` Pydantic model fields. Verify `ErrorResponse` and
`OpenAIBaseModel` are still at `engine.protocol`. Check `validate_json_request` still
returns a FastAPI `Depends`.

______________________________________________________________________

### 4. `vllm.entrypoints.utils` — server utilities

**Source:** `vllm/entrypoints/utils.py`

Called in `areal/engine/vllm_ext/areal_vllm_server.py` (lines 19, 349-350, 390):

```python
from vllm.entrypoints.utils import cli_env_setup, load_aware_call, with_cancellation

# Used as stacked decorators on the create_completion endpoint (lines 349-350):
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    ...

# Called at server startup (line 390):
cli_env_setup()
```

**Check:** Confirm all three are still exported from `vllm.entrypoints.utils`. Verify
`with_cancellation` is still a decorator compatible with async endpoints. Verify
`load_aware_call` is still a decorator (not renamed to `load_aware`). Confirm
`cli_env_setup()` is still a no-arg function.

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

______________________________________________________________________

## Version-Guarded Code

- `areal/infra/platforms/cuda.py` and `unknown.py` — tries
  `vllm.v1.worker.gpu_worker.Worker` first, falls back to `vllm.worker.worker.Worker`.
  Once V0 support is dropped, remove the fallback.

- `asystem-awex/awex/awex_vllm_server.py` — `FlexibleArgumentParser` wrapped in
  try/except with fallback to `argparse.ArgumentParser`.

- `asystem-awex/awex/vllm_plugin.py` — `get_pcp_group` import wrapped in try/except
  (added in newer vLLM versions).
