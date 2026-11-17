# CPU vision memory

Long multimodal trajectories can exhaust host memory while GPU memory remains available.
These changes address three separate allocation lifetimes:

- **Image sharing:** preserve immutable image aliases through RTensor token compaction,
  and optionally broadcast CPU vision tensors into one mapped backing store per node
  instead of allocating a copy on every model-parallel rank.
- **RPC encoding:** encode a contiguous NumPy buffer directly, avoiding a full temporary
  `bytes` copy before Base64 encoding. The wire format is unchanged, including the
  existing BF16-to-FP32 wire representation.
- **Historical image decoding:** decode full-resolution Qwen images inside the per-image
  resize cache factory. Cache hits reuse resized images without decoding the originals
  again. Consumers still receive independent PIL copies.

## Enable node-local sharing

Set `AREAL_NODE_SHARED_VISION=1` consistently on every actor rank before RPC calls. The
default is disabled. Local ranks must share a writable mapping directory, selected by
`AREAL_VISION_SHM_DIR` (default `/dev/shm`). Provision enough shared memory for the live
vision tensors; the mapping remains charged to host/cgroup memory after unlinking its
filename.

Sharing applies to alias-preserving CPU vision payloads on Gloo groups, including
independently scheduled, disjoint DP groups. Each DP group owns separate mappings on
each node: different DP batches never share image storage. Metadata collectives and
leader-to-leader pixel transfers reuse the existing parent group; no additional process
groups are created, and unrelated DP ranks need not participate. Calls within each
parent group must remain ordered. Non-Gloo groups use the original broadcast. The source
retains its original input in addition to the shared mapping. Image tensors must be
treated as immutable by consumers; token tensors retain their normal behavior. Pixel
transfers fan out from the source to one leader per destination node; this preserves
memory savings but is not a claim of improved throughput at large node counts.

Mappings are unlinked after all local ranks open them, before copying data. Their
lifetime is owned by live tensor references. Do not delete active shared-memory mappings
as an OOM workaround. This feature does not increase cgroup limits or release
model/optimizer memory.

## Compare a single workload

`tests/benchmarks/vision_memory.py` runs a synthetic proxy-to-actor pipeline in fresh
processes: two prompts reuse the same eight images, PIL resizes 2048-square images to
512-square images, and 12 trajectories reference the same 24 MiB float32 vision tensor.
Actual RTensor compaction uses an in-memory storage adapter. The unique stored image
shards are encoded and transferred through a pipe, and eight real Gloo ranks receive
them. A separate sampler reads the proxy and all actor processes' PSS approximately
every 20 ms plus sampling overhead. Shared pages are counted proportionally instead of
summing duplicate RSS.

Five variants use identical input, sizes and rank counts:

| Variant       | Alias fix + node sharing | Direct RPC buffer | Lazy decode |
| ------------- | ------------------------ | ----------------- | ----------- |
| `main`        | No                       | No                | No          |
| `shared`      | Yes                      | No                | No          |
| `shared-rpc`  | Yes                      | Yes               | No          |
| `shared-lazy` | Yes                      | No                | Yes         |
| `all`         | Yes                      | Yes               | Yes         |

All variants must produce the same output SHA256. Main produces 12 distinct image shards
during compaction; all sharing variants produce one. `original_decodes` should be 16
without lazy decode and 8 with it. The benchmark fixes the image policy explicitly and
uses dense RGB float32 storage; it does not invoke a model-specific HF processor, HTTP
server, GPU model or optimizer. The storage adapter retains actual tensors without
networking; token shard transfers are excluded. The benchmark therefore measures the
combined image-processing/compaction/encoding/broadcast path, not a production training
run.

Export the unchanged serializer from the tested main revision and run each variant in a
fresh interpreter, from the repository root with its dependencies installed:

```bash
baseline_dir=$(mktemp -d)
git show 7fb167a30:areal/infra/rpc/serialization.py > "$baseline_dir/serialization.py"
git show 7fb167a30:areal/infra/rpc/rtensor.py > "$baseline_dir/rtensor.py"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
for variant in main shared shared-rpc shared-lazy all; do
  python tests/benchmarks/vision_memory.py \
    --variant "$variant" --baseline-root "$baseline_dir" \
    --output "/tmp/vision-memory-$variant.json"
done
```

Compare both `peak_pss_mib` (maximum simultaneous aggregate PSS) and the `resident`
phase after broadcasting. Repeat runs and report ranges: process initialization,
allocator retention and PSS sampling cause variation. Phase peaks cannot be added:
reducing encoding allocations may leave the JSON serialization/transfer stage as the
global peak. Likewise, the node-shared source still holds its original tensor. The
sampler excludes its own process and samples processes sequentially, so these are
sampled estimates rather than atomic kernel high-water marks.

## Measured example

Measured on 2026-11-17 against main `7fb167a30`, using Python 3.12.3, PyTorch 2.9.1, the
default workload above and three fresh runs per variant. All 15 runs produced identical
image hashes. Units below are MiB.

| Variant       | Median total peak PSS | Three-run range |
| ------------- | --------------------: | --------------: |
| `main`        |                7790.7 |   7779.8–7816.3 |
| `shared`      |                5345.8 |   5341.3–5349.3 |
| `shared-rpc`  |                5350.3 |   5335.7–5365.7 |
| `shared-lazy` |                5278.4 |   5263.3–5278.9 |
| `all`         |                5279.4 |   5238.9–5286.7 |

The combined reduction is 2511.4 MiB (32.2%). Most comes from sharing; adding both copy
optimizations to sharing reduces this workload's median total peak by 66.5 MiB (1.2%).
The RPC-only incremental difference is within run variation: avoiding an encoding
allocation does not guarantee a reduction in the maximum of later JSON/transfer phases.
Lazy decode accounts for the measurable additional reduction here. Do not add isolated
microbenchmark savings or extrapolate these percentages to a full training job.

The initial version passed 173 targeted CPU tests and a real two-node/four-rank Gloo
run. The DP-capable transport passes 175 targeted CPU tests. Additional real
two-node/eight-rank tests cover both node-local DP replicas and two DP replicas that
each span both nodes. They verify staggered startup, unequal stream lengths, changing
source ranks, singleton groups, repeated aliases, actual shared storage, isolation
between replicas and nodes, and removal of mapping filenames. These are CPU transport
tests; they do not validate a full GPU training run or establish DP-specific savings
percentages.
