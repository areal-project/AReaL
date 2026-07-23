#!/usr/bin/env python3
"""
Patch: Enable Bailing Hybrid model radix cache support in SGLang 0.5.9
Aligned with the Ling hybrid SGLang fork behavior

Modifies 8 files:
  1. server_args.py                       — Enable mamba cache for BailingMoeV2_5ForCausalLM
  2. scheduler.py                         — Add hybrid_lightning_config to is_hybrid_ssm
  3. dp_attention.py                      — Add force_sum_len_attn_dp for DP attention mode
  4. model_runner.py                      — Pass force_sum_len_attn_dp for BailingHybrid models
  5. forward_batch_deepseek_mha_mixin.py  — Support HybridLinearKVPool in chunked prefix cache
  6. common.py                            — Fix mamba eviction factor for no_buffer mode
  7. memory_pool.py                       — Clear stale Mamba extra-buffer request state on free
  8. scheduler_runtime_checker_mixin.py   — Exclude Mamba dummy slot 0 from leak diagnostics

Usage:
  python apply_sgl_radix_cache_patch.py
  python apply_sgl_radix_cache_patch.py --sglang-path /path/to/sglang
  python apply_sgl_radix_cache_patch.py --dry-run
  python apply_sgl_radix_cache_patch.py --revert
"""

import argparse
import os
import shutil
import subprocess
import sys

PATCHES = [
    # ── 1. server_args.py ──
    {
        "name": "1. server_args: BailingMoeV2_5 support_mamba_cache -> True",
        "rel_path": os.path.join("srt", "server_args.py"),
        "old": (
            '        elif model_arch in ["KimiLinearForCausalLM", "BailingMoeV2_5ForCausalLM"]:\n'
            "            self._handle_mamba_radix_cache(\n"
            "                model_arch=model_arch,\n"
            "                support_mamba_cache=False,\n"
            "            )"
        ),
        "new": (
            '        elif model_arch in ["KimiLinearForCausalLM"]:\n'
            "            self._handle_mamba_radix_cache(\n"
            "                model_arch=model_arch,\n"
            "                support_mamba_cache=False,\n"
            "            )\n"
            '        elif model_arch in ["BailingMoeV2_5ForCausalLM"]:\n'
            "            self._handle_mamba_radix_cache(\n"
            "                model_arch=model_arch,\n"
            "                support_mamba_cache=True,\n"
            "                support_mamba_cache_extra_buffer=True,\n"
            "            )"
        ),
        "check": 'elif model_arch in ["BailingMoeV2_5ForCausalLM"]:',
    },
    # ── 2. scheduler.py ──
    {
        "name": "2. scheduler: is_hybrid_ssm += hybrid_lightning_config",
        "rel_path": os.path.join("srt", "managers", "scheduler.py"),
        "old": (
            "        self.is_hybrid_ssm = (\n"
            "            self.tp_worker.model_runner.hybrid_gdn_config is not None\n"
            "            or self.tp_worker.model_runner.mamba2_config is not None\n"
            "        )"
        ),
        "new": (
            "        self.is_hybrid_ssm = (\n"
            "            self.tp_worker.model_runner.hybrid_gdn_config is not None\n"
            "            or self.tp_worker.model_runner.mamba2_config is not None\n"
            "            or self.tp_worker.model_runner.hybrid_lightning_config is not None\n"
            "        )"
        ),
        "check": "hybrid_lightning_config is not None",
    },
    # ── 3a. dp_attention.py: _USE_SUM_LEN_ATTN_DP global ──
    {
        "name": "3a. dp_attention: add _USE_SUM_LEN_ATTN_DP global",
        "rel_path": os.path.join("srt", "layers", "dp_attention.py"),
        "old": "class DpPaddingMode(IntEnum):",
        "new": (
            "_USE_SUM_LEN_ATTN_DP: bool = False\n\n\nclass DpPaddingMode(IntEnum):"
        ),
        "check": "_USE_SUM_LEN_ATTN_DP: bool = False",
    },
    # 3b: check in get_dp_padding_mode
    {
        "name": "3b. dp_attention: check _USE_SUM_LEN_ATTN_DP in get_dp_padding_mode",
        "rel_path": os.path.join("srt", "layers", "dp_attention.py"),
        "old": (
            "    @classmethod\n"
            "    def get_dp_padding_mode(\n"
            "        cls, is_extend_in_batch, global_num_tokens: List[int]\n"
            "    ) -> DpPaddingMode:\n"
            "        if is_extend_in_batch:"
        ),
        "new": (
            "    @classmethod\n"
            "    def get_dp_padding_mode(\n"
            "        cls, is_extend_in_batch, global_num_tokens: List[int]\n"
            "    ) -> DpPaddingMode:\n"
            "        if _USE_SUM_LEN_ATTN_DP:\n"
            "            return DpPaddingMode.SUM_LEN\n"
            "\n"
            "        if is_extend_in_batch:"
        ),
        "check": "_USE_SUM_LEN_ATTN_DP:\n            return DpPaddingMode.SUM_LEN",
    },
    # 3c: add param to initialize_dp_attention
    {
        "name": "3c. dp_attention: add force_sum_len_attn_dp param",
        "rel_path": os.path.join("srt", "layers", "dp_attention.py"),
        "old": (
            "def initialize_dp_attention(\n"
            "    server_args: ServerArgs,\n"
            "    model_config: ModelConfig,\n"
            "):\n"
            "    global _ATTN_DP_RANK, _ATTN_DP_SIZE\n"
            "    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG\n"
            "    enable_dp_attention = server_args.enable_dp_attention"
        ),
        "new": (
            "def initialize_dp_attention(\n"
            "    server_args: ServerArgs,\n"
            "    model_config: ModelConfig,\n"
            "    force_sum_len_attn_dp: bool = False,\n"
            "):\n"
            "    global _ATTN_DP_RANK, _ATTN_DP_SIZE\n"
            "    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG\n"
            "    global _USE_SUM_LEN_ATTN_DP\n"
            "    enable_dp_attention = server_args.enable_dp_attention\n"
            "    _USE_SUM_LEN_ATTN_DP = force_sum_len_attn_dp"
        ),
        "check": "force_sum_len_attn_dp: bool = False",
    },
    # 3d: add is_use_sum_len_attn_dp function
    {
        "name": "3d. dp_attention: add is_use_sum_len_attn_dp()",
        "rel_path": os.path.join("srt", "layers", "dp_attention.py"),
        "old": (
            "def is_dp_attention_enabled() -> bool:\n"
            "    return _ENABLE_DP_ATTENTION_FLAG\n"
            "\n"
            "\n"
            "def is_allocation_symmetric()"
        ),
        "new": (
            "def is_dp_attention_enabled() -> bool:\n"
            "    return _ENABLE_DP_ATTENTION_FLAG\n"
            "\n"
            "\n"
            "def is_use_sum_len_attn_dp() -> bool:\n"
            "    return _USE_SUM_LEN_ATTN_DP\n"
            "\n"
            "\n"
            "def is_allocation_symmetric()"
        ),
        "check": "def is_use_sum_len_attn_dp",
    },
    # ── 4. model_runner.py ──
    {
        "name": "4. model_runner: pass force_sum_len_attn_dp",
        "rel_path": os.path.join("srt", "model_executor", "model_runner.py"),
        "old": (
            "            initialize_dp_attention(\n"
            "                server_args=self.server_args,\n"
            "                model_config=self.model_config,\n"
            "            )"
        ),
        "new": (
            "            initialize_dp_attention(\n"
            "                server_args=self.server_args,\n"
            "                model_config=self.model_config,\n"
            "                force_sum_len_attn_dp=True if self.hybrid_lightning_config else False,\n"
            "            )"
        ),
        "check": "force_sum_len_attn_dp=True if self.hybrid_lightning_config",
    },
    # ── 5. forward_batch_deepseek_mha_mixin.py: support HybridLinearKVPool for chunked prefix cache ──
    {
        "name": "5. chunked_prefix_cache: support HybridLinearKVPool",
        "rel_path": os.path.join(
            "srt", "model_executor", "forward_batch_deepseek_mha_mixin.py"
        ),
        "old": (
            "        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool\n"
            "\n"
            "        assert isinstance(\n"
            "            self.token_to_kv_pool, MLATokenToKVPool\n"
            '        ), "Currently chunked prefix cache can only be used by Deepseek models"'
        ),
        "new": (
            "        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MLATokenToKVPool\n"
            "\n"
            "        assert isinstance(\n"
            "            self.token_to_kv_pool, (MLATokenToKVPool, HybridLinearKVPool)\n"
            '        ), "Chunked prefix cache requires MLATokenToKVPool or HybridLinearKVPool"'
        ),
        "check": "MLATokenToKVPool, HybridLinearKVPool",
    },
    # ── 6. common.py: fix mamba eviction factor for no_buffer mode ──
    {
        "name": "6. common: fix mamba eviction factor (3x -> 1x for no_buffer)",
        "rel_path": os.path.join("srt", "mem_cache", "common.py"),
        "old": (
            "    if isinstance(req_to_token_pool, HybridReqToTokenPool):\n"
            "        mamba_available_size = req_to_token_pool.mamba_pool.available_size()\n"
            "        factor = (\n"
            "            MAMBA_STATE_PER_REQ_PREFIX_CACHE\n"
            "            if tree_cache.supports_mamba()\n"
            "            else MAMBA_STATE_PER_REQ_NO_CACHE\n"
            "        )\n"
            "        mamba_state_needed = num_reqs * factor"
        ),
        "new": (
            "    if isinstance(req_to_token_pool, HybridReqToTokenPool):\n"
            "        mamba_available_size = req_to_token_pool.mamba_pool.available_size()\n"
            "        factor = (\n"
            "            MAMBA_STATE_PER_REQ_PREFIX_CACHE\n"
            "            if tree_cache.supports_mamba()\n"
            "            and getattr(tree_cache, 'enable_mamba_extra_buffer', False)\n"
            "            else MAMBA_STATE_PER_REQ_NO_CACHE\n"
            "        )\n"
            "        mamba_state_needed = num_reqs * factor"
        ),
        "check": "getattr(tree_cache, 'enable_mamba_extra_buffer', False)",
    },
    # ── 7. memory_pool.py: backport #26941 mamba_extra_buffer ping-pong slot leak fix ──
    {
        "name": "7. memory_pool: clear stale mamba extra-buffer req state",
        "rel_path": os.path.join("srt", "mem_cache", "memory_pool.py"),
        "old": "            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)",
        "new": (
            "            self.mamba_pool.free(mamba_ping_pong_track_buffer_to_free)\n"
            "            # Match req.mamba_pool_idx=None above so the next alloc does not\n"
            "            # see a stale ping-pong reference and skip allocating fresh slots.\n"
            "            req.mamba_ping_pong_track_buffer = None\n"
            "            req.mamba_next_track_idx = None"
        ),
        "check": "req.mamba_ping_pong_track_buffer = None",
    },
    # ── 8. scheduler_runtime_checker_mixin.py: dummy slot 0 is reserved ──
    {
        "name": "8. checker: exclude mamba dummy slot 0",
        "rel_path": os.path.join(
            "srt", "managers", "scheduler_runtime_checker_mixin.py"
        ),
        "old": "            expected_mamba_pages = set(range(self.req_to_token_pool.mamba_pool.size))",
        "new": (
            "            expected_mamba_pages = set(\n"
            "                range(1, self.req_to_token_pool.mamba_pool.size + 1)\n"
            "            )"
        ),
        "check": "range(1, self.req_to_token_pool.mamba_pool.size + 1)",
    },
]


def find_sglang_root():
    try:
        import sglang

        return os.path.dirname(sglang.__file__)
    except ImportError:
        return None


def resolve(root, rel):
    p = os.path.join(root, rel)
    if os.path.exists(p):
        return p
    p2 = os.path.join(root, "python", "sglang", rel)
    return p2 if os.path.exists(p2) else p


def apply_patch(root, patch, dry_run=False):
    fp = resolve(root, patch["rel_path"])
    if not os.path.exists(fp):
        return "MISSING", f"File not found: {fp}"
    with open(fp) as f:
        content = f.read()
    if patch["check"] in content:
        return "SKIP", "Already patched"
    if patch["old"] not in content:
        return "FAIL", f"Old code not found in {os.path.basename(fp)}"
    if dry_run:
        return "DRY", f"Would patch {os.path.basename(fp)}"
    bak = fp + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(fp, bak)
    new_content = content.replace(patch["old"], patch["new"], 1)
    with open(fp, "w") as f:
        f.write(new_content)
    with open(fp) as f:
        if patch["check"] in f.read():
            return "OK", f"Patched {os.path.basename(fp)}"
    return "FAIL", f"Verify failed for {os.path.basename(fp)}"


def revert_patch(root, patch):
    fp = resolve(root, patch["rel_path"])
    if not os.path.exists(fp):
        return "SKIP", "File not found"
    bak = fp + ".bak"
    if os.path.exists(bak):
        shutil.copy2(bak, fp)
        os.remove(bak)
        return "OK", "Restored from backup"
    with open(fp) as f:
        content = f.read()
    if patch["new"] in content:
        content = content.replace(patch["new"], patch["old"], 1)
        with open(fp, "w") as f:
            f.write(content)
        return "OK", "Reverted inline"
    return "SKIP", "Not patched"


def clear_pycache(root):
    subprocess.run(
        [
            "find",
            root,
            "-name",
            "__pycache__",
            "-type",
            "d",
            "-exec",
            "rm",
            "-rf",
            "{}",
            "+",
        ],
        capture_output=True,
    )
    subprocess.run(["find", root, "-name", "*.pyc", "-delete"], capture_output=True)


def main():
    parser = argparse.ArgumentParser(
        description="Patch SGLang for Bailing Hybrid radix cache"
    )
    parser.add_argument("--sglang-path", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("  Bailing Hybrid Radix Cache Patch")
    print("=" * 64)

    root = args.sglang_path or find_sglang_root()
    if root is None:
        print("\n[ERROR] Cannot find sglang. Use --sglang-path.")
        sys.exit(1)
    print(f"\n  SGLang root: {root}\n")

    all_ok = True
    for p in PATCHES:
        if args.revert:
            status, msg = revert_patch(root, p)
        else:
            status, msg = apply_patch(root, p, args.dry_run)
        icon = {"OK": "V", "SKIP": "~", "DRY": "?", "FAIL": "X", "MISSING": "X"}[status]
        print(f"  [{icon}] {p['name']}")
        if status in ("FAIL", "MISSING"):
            print(f"      {msg}")
            all_ok = False

    if not args.dry_run and not args.revert:
        clear_pycache(root)
        print("\n  [V] Cleared __pycache__")

    print("\n" + "=" * 64)
    if all_ok:
        if args.dry_run:
            print("  Dry run OK.")
        elif args.revert:
            print("  Reverted.")
        else:
            print("  All patches applied!")
            print("\n  Changes:")
            print("    1. BailingMoeV2_5 radix cache: DISABLED -> ENABLED")
            print("    2. Scheduler: is_hybrid_ssm includes hybrid_lightning_config")
            print("    3. DP attention: force_sum_len_attn_dp support")
            print("    4. Model runner: pass force_sum_len for BailingHybrid")
            print("    5. Chunked prefix cache: support HybridLinearKVPool")
            print("    6. Mamba eviction: factor 3x -> 1x for no_buffer mode")
            print("    7. Mamba extra_buffer: clear stale ping-pong request state")
            print("    8. Mamba checker: exclude reserved dummy slot 0")
            print("\n  Revert: python apply_sgl_radix_cache_patch.py --revert")
    else:
        print("  Some patches failed.")
        sys.exit(1)
    print("=" * 64)


if __name__ == "__main__":
    main()
