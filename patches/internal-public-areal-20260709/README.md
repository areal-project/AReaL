# Internal public AReaL hotfix archive (2026-07-09)

Diffs applied directly to /storage/openpsi/users/public/projects/AReaL
(the tree used by xrd-compaction colocate runs, e.g. job 939460) so recover
works on colocated AWEX trials. Pre-patch originals are archived at
/tmp/opencode/internal-public-AReaL-backup-20260709-161019 on the login node.

- recover.diff: AWEX-colocate recover ordering — pause -> pause_generation_sync
  -> offload kv_cache/weights -> load actor checkpoint -> update_weights, with
  resume in a finally block. Equivalent logic on this branch: f5f48cb5.
- remote_inf_engine.diff: fail-fast when the local inference server process
  dies before its health check, plus a 60s waiting heartbeat. Equivalent logic
  on this branch: bf9ab455 (signature: _wait_for_server(address, process=None)).

Verified by job 939460: recover passed engine init, offloaded rollout before
DCP checkpoint load, published 64 writer versions, and resumed sampling.
