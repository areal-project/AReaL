# Prefill-Decode Disaggregation

Run a rollout with **Prefill-Decode (PD) disaggregation** through the AReaL inference
service. PD splits an SGLang deployment into a prefill group and a decode group; each
chat request runs prefill on one server, then streams the KV cache to a decode server.

Hardware: 2 GPUs (1 prefill + 1 decode), currently tested on TP1 PP1.

## Install a KV transport engine

AReaL does not bundle one. Install one yourself:

```bash
pip install mooncake-transfer-engine    # default, supports RDMA and TCP
# or
pip install nixl                         # alternative, UCX-based
```

## Run

```bash
python3 examples/experimental/inference_service/online_rollout.py \
    --config examples/experimental/inference_service/online_rollout.yaml \
    rollout.backend="sglang(P:d1t1p1|D:d1t1p1)" \
    cluster.n_gpus_per_node=2 \
    actor.path=Qwen/Qwen3-0.6B
```

See [Online Rollout](online_proxy.md) for the non-PD baseline.
