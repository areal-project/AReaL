# Archon 测试说明

## `test_dta.py` 简介

`test_dta.py` 主要验证 Archon 的 DTA 路径，包括：

- `forward_batch` 冒烟检查
- `train_batch` 冒烟检查
- 与 FSDP 的数值一致性对比

## 测试函数说明

- `test_engine_is_initialized`：检查引擎能否正常初始化，并确认 DTA 开关状态正确。
- `test_forward_batch_runs`：只验证 Archon 的 `forward_batch` 在 DTA 开启时可正常跑通。
- `test_train_batch_runs`：只验证 Archon 的 `train_batch` 在 DTA 开启时可正常跑通并返回结果。
- `test_forward_batch_matches_fsdp`：对比 Archon 与 FSDP 的 `forward_batch`
  输出，检查形状和数值误差是否在可接受范围内。
- `test_train_batch_matches_fsdp`：对比 Archon 与 FSDP
  一次训练步后的梯度范数和参数更新量，检查训练信号一致性。很难强对齐，建议观察 grad_norm 和 delta_norm 是否对齐。

## 输入数据格式

通过 `--dta-data` 传入一个 `.pt` 文件，内容要求：

- 类型是 `list[torch.Tensor]`
- 每个元素是 1-D token 序列（不做 padding）

示例：

```python
[
    torch.tensor([101, 2023, 2003, 1037, 3231], dtype=torch.long),
    torch.tensor([101, 2064, 2017, 2393, 1029], dtype=torch.long),
]
```

## 参数说明

- `--dta-data PATH`：DTA 数据文件路径；不传会跳过 DTA 测试
- `--dta-limit INT`：最多使用前 N 条序列，`-1` 表示全部使用
- `--max-tokens-per-mb INT`：单条序列token 上限（用于序列/微批控制）
- `--no-dta`：关闭 DTA
- `--use-hf`：model 使用 HuggingFace 模型路径分支，即去掉 archon 包装
- `--model-path PATH`：模型路径（与 `--use-hf` 搭配）

## 用法示例（`python -m pytest`）

只跑 DTA 测试：

```bash
python -m pytest -v -s tests/experimental/archon/test_dta.py \
  --dta-data /path/to/dta_samples.pt
```

限制样本数量（快速迭代）：

```bash
python -m pytest -v -s tests/experimental/archon/test_dta.py \
  --dta-data /path/to/dta_samples.pt \
  --dta-limit 16
```

调整 token 上限：

```bash
python -m pytest -v -s tests/experimental/archon/test_dta.py \
  --dta-data /path/to/dta_samples.pt \
  --max-tokens-per-mb 4096
```

使用 HF 模型路径：

```bash
python -m pytest -v -s tests/experimental/archon/test_dta.py \
  --dta-data /path/to/dta_samples.pt \
  --use-hf \
  --model-path /path/to/model
```

按函数精确运行（`::`）：

```bash
python -m pytest -v -s tests/experimental/archon/test_dta.py::test_forward_batch_runs \
  --dta-data /path/to/dta_samples.pt
```

```bash
python -m pytest -v -s tests/experimental/archon/test_dta.py::test_train_batch_matches_fsdp \
  --dta-data /path/to/dta_samples.pt
```
