# Sequence Packing Algorithms

AReaL supports configurable sequence packing algorithms for micro-batch
allocation during training. Sequence packing determines how variable-length
sequences are grouped into micro-batches, directly impacting load balance
across data-parallel (DP) ranks and overall training throughput.

## Supported Algorithms

| Algorithm | Key | Description | Complexity | Balance Quality |
|---|---|---|---|---|
| **First Fit Decreasing (FFD)** | `ffd` | Greedy bin-packing heuristic. Sorts sequences by length (descending) and assigns each to the first bin with remaining capacity. | O(n log n) | Good |
| **Karmarkar-Karp (KK)** | `kk` | Largest Differencing Method. Iteratively merges the two most imbalanced partial partitions using a max-heap, producing near-optimal balance. | O(n log n · k) | Excellent |

## Configuration

The packing algorithm is controlled by the `packing_algorithm` field in
`MicroBatchSpec`, which can be set directly in YAML configuration files.

### YAML Configuration

```yaml
# In your experiment config (e.g., examples/countdown/train_config.yaml)

actor:
  mb_spec:
    max_tokens_per_mb: 8192
    n_mbs: 4
    n_mbs_divisor: 1
    packing_algorithm: kk    # Options: "ffd" (default), "kk"
```

### Python API

You can also set the algorithm programmatically:

```python
from areal.api.cli_args import MicroBatchSpec

# Using KK algorithm
mb_spec = MicroBatchSpec(
    max_tokens_per_mb=8192,
    n_mbs=4,
    packing_algorithm="kk",
)

# Or update an existing spec
mb_spec_kk = MicroBatchSpec.new(existing_spec, packing_algorithm="kk")
```

## When to Use KK

**Recommended scenarios for KK:**

- **Large-scale RL training** with highly variable sequence lengths (e.g., RLHF,
  PPO with open-ended generation). KK significantly reduces the spread between
  the most-loaded and least-loaded DP rank.
- **Bimodal sequence distributions** where a mix of very short and very long
  sequences makes greedy packing suboptimal.
- **High DP parallelism** (≥4 ranks), where even small load imbalances cause
  significant idle time due to synchronization barriers.

**When FFD is sufficient:**

- Uniform or near-uniform sequence lengths.
- Small-scale experiments where packing overhead matters more than balance.
- Latency-sensitive inference pipelines (FFD is slightly faster).
