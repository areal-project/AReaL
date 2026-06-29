# Prefill-Decode 解耦

通过 AReaL 推理服务运行 **Prefill-Decode (PD) 解耦** rollout。PD 将 SGLang 拆成 prefill 组和 decode 组：每条
chat 请求先在 prefill 上做 prompt 预填充，再把 KV cache 传给 decode 完成生成。

硬件：2 张 GPU（1 prefill + 1 decode），目前只在 TP1 PP1 上验证过。

## 安装 KV 传输引擎

AReaL 不再默认安装，请自行选择：

```bash
pip install mooncake-transfer-engine    # 默认，支持 RDMA 和 TCP
# 或
pip install nixl                         # 基于 UCX 的替代方案
```

## 运行

```bash
python3 examples/experimental/inference_service/online_rollout.py \
    --config examples/experimental/inference_service/online_rollout.yaml \
    'rollout.backend="sglang(P:d1t1p1|D:d1t1p1)"' \
    cluster.n_gpus_per_node=2 \
    actor.path=Qwen/Qwen3-0.6B
```

非 PD 基线参见 [Online Rollout](online_proxy.md)。
