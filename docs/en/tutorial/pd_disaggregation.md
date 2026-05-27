# Prefill-Decode Disaggregation

This page is a quickstart for running rollouts with **Prefill-Decode (PD)
disaggregation** through the AReaL inference service. PD splits an SGLang deployment
into a prefill group and a decode group; each chat request runs its prompt prefill on
the prefill server, then streams the KV cache to the decode server which generates
tokens.

## When to use it

- The model and traffic pattern make a single co-located server inefficient (large
  models, very long prompts, or imbalanced prefill/decode load).
- You have at least 2 GPUs and a working KV-cache transport between them.

## Install a KV transport engine

AReaL does not bundle any KV-cache transfer engine. Install one yourself, matching your
network:

```bash
# Option A: Mooncake (default, supports RDMA and TCP)
pip install mooncake-transfer-engine

# Option B: NIXL (alternative, uses UCX)
pip install nixl
```

Both ship Linux x86_64 wheels only.

## Configure

Set two fields in your rollout config (everything else is the same as a regular online
rollout):

```yaml
rollout:
  _version: v2
  backend: "sglang:d2"      # DP=2: group 0 = prefill, group 1 = decode. Currently only tested on TP1PP1.
  pd_disaggregation: true
```

## Run

`pd_online_rollout.py` reuses the existing `online_rollout.yaml` and toggles PD via CLI
overrides:

```bash
python3 examples/experimental/inference_service/pd_online_rollout.py \
    --config examples/experimental/inference_service/online_rollout.yaml \
    rollout.backend="sglang:d2" \
    rollout.pd_disaggregation=true \
    cluster.n_gpus_per_node=2 \
    actor.path=Qwen/Qwen3-0.6B
```

The log prints the gateway URL plus the prefill and decode addresses:

```
InferenceServicePDOnlineTrain INFO: Proxy gateway available at http://127.0.0.1:<PORT>
InferenceServicePDOnlineTrain INFO: PD prefill addrs: ['http://...']
InferenceServicePDOnlineTrain INFO: PD decode addrs:  ['http://...']
```

## See also

- [Online Rollout](online_proxy.md) — the non-PD baseline that PD layers on top of.
- `examples/experimental/inference_service/README.md` — full README for both examples.
