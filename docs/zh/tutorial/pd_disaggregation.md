# Prefill-Decode 解耦

本页是通过 AReaL 推理服务运行 **Prefill-Decode (PD) 解耦** rollout 的快速指南。 PD 将 SGLang 部署拆成 prefill 组和
decode 组：每条 chat 请求先在 prefill 服务器上做 prompt 预填充，再把 KV cache 传给 decode 服务器完成生成。

## 何时启用

- 模型或流量模式在单服务器上效率低（大模型、超长 prompt 或 prefill/decode 负载严重失衡）。
- 至少有 2 张 GPU，且这两张 GPU 之间有可用的 KV cache 传输通道。

## 安装KV传输引擎

AReaL 默认不包含任何 KV cache 传输引擎。请根据网络环境自行安装一个：

```bash
# 选项 A：Mooncake（默认，支持 RDMA 和 TCP等多种方案）
pip install mooncake-transfer-engine

# 选项 B：NIXL（基于 UCX 的替代方案）
pip install nixl
```

## 配置

在 rollout 配置中设置两个字段（其余与常规 online rollout 相同）：

```yaml
rollout:
  _version: v2
  backend: "sglang:d2"      # DP=2：group 0 = prefill, group 1 = decode. 目前仅测试TP1PP1架构.
  pd_disaggregation: true
```

## 运行

`pd_online_rollout.py` 复用现有的 `online_rollout.yaml`，通过命令行 override 切到 PD：

```bash
python3 examples/experimental/inference_service/pd_online_rollout.py \
    --config examples/experimental/inference_service/online_rollout.yaml \
    rollout.backend="sglang:d2" \
    rollout.pd_disaggregation=true \
    cluster.n_gpus_per_node=2 \
    actor.path=Qwen/Qwen3-0.6B
```

日志会打印 gateway URL 以及 prefill / decode 地址：

```
InferenceServicePDOnlineTrain INFO: Proxy gateway available at http://127.0.0.1:<PORT>
InferenceServicePDOnlineTrain INFO: PD prefill addrs: ['http://...']
InferenceServicePDOnlineTrain INFO: PD decode addrs:  ['http://...']
```

## 参考

- [Online Rollout](online_proxy.md) —— PD 在其之上叠加的非 PD 基线。
- `examples/experimental/inference_service/README.md` —— 两个 example 的完整 README。
