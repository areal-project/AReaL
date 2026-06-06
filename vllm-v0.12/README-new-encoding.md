# New encoding for testtime scaling

## Installation

Refer to vllm's official site. But it must be built from source. A feasible choice is

1. Create virtual env
```
python3 -n venv .vllm
source .vllm/bin/activate
```
2. Install the package
```
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -i (ant group's pip image)
```
The -i should be added otherwise there is netork error

It takes about 10 mins

## Start server
```
python -m vllm.entrypoints.openai.api_server     --model /storage/openpsi/models/Qwen__Qwen3-4B      --hf-overrides '{"rope_parameters": {"rope_type": "chunked"}}' &
```
Now there is a server working at port 8000

## Run inference

```
python2 vllm_existing_server.py
```