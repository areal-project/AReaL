# Preserve frozen draft state during colocated memory release

Native draft CPU backup tracks allocations. CUDA allocator reuse can place independent
speculative parameters and derived buffers in allocations without draft backup. A
release/resume cycle can therefore zero draft norms despite the backup flag.

Preserve independent draft parameters and buffers on CPU before the first effective
weight release. Restore them in place after native weight resume, before generation
continues. Exclude any storage shared with the target model so embedding/head updates
remain visible. Validate tensor identity and storage addresses before restoring to
protect captured CUDA graphs. Keep the snapshot across cycles because this path uses a
frozen draft; training the draft would require a separate refresh contract.

Integrate at the existing AWEX scheduler plugin's native memory transition wrappers,
including cached request dispatch callbacks. Adapter-only interception would miss native
HTTP release requests. Keep existing tag filtering and duplicate handling. KV-only
transitions do not allocate or restore a snapshot.

Relying solely on the native backup flag is insufficient given observed zero norms.
Reloading draft weights from disk each cycle adds repeated I/O and requires remapping
shared weights. A one-time CPU snapshot follows the previously verified implementation.

Validation covers repeated cycles, nonpersistent buffers, shared storage aliases,
replacement detection, direct/nested draft runners, native request dispatch and separate
weight-updater objects. An actual-model canary checks independent draft checksums across
native memory-saver cycles with CUDA graphs enabled. Subsequent RL recovery must
separately prove target synchronization, healthy MTP acceptance, new optimizer updates,
numerical metrics and full checkpoint publication.
