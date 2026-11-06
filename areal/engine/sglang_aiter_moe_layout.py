# SPDX-License-Identifier: Apache-2.0
"""Keep aiter-shuffled SGLang MoE weights correct across online weight updates.

On ROCm with ``SGLANG_USE_AITER=1``, SGLang tile-shuffles every unquantized
FusedMoE ``w13_weight`` / ``w2_weight`` once in ``process_weights_after_loading``.
``update_weights_from_tensor`` / ``update_weights_from_distributed`` only call
``load_weights``, whose weight loader writes checkpoint-layout rows into those
shuffled buffers; every MoE layer then computes garbage without raising.

The patch makes the weight loader unshuffle the target expert, load it, and
shuffle it back, in place. ``Dockerfile.rocm`` applies it at build time; on
other ROCm installations run ``python -m areal.engine.sglang_aiter_moe_layout``
once before starting SGLang. Only the validated SGLang 0.5.10.post1 sources are
accepted. SGLang >= 0.5.21 re-runs post-load processing in its weight-update
sessions instead; this patch does not apply there.
"""

import argparse
from pathlib import Path

from areal.engine._sglang_source_patch import (
    install_source_patches,
    installed_sglang_root,
    patch_source,
)

LAYER_RELATIVE = Path("srt/layers/moe/fused_moe_triton/layer.py")
LAYER_EXPECTED_SHA256 = (
    "fc7ff335440a4901d4756153e1225d207f67f392a33a313fcb7ac1a896623afe"
)
LAYER_PATCHED_SHA256 = (
    "80bb66ca5580dddcc9e3763b8fe92f9afbfeebd9a66594c703c3803872a90334"
)
LAYER_REPLACEMENTS = [
    (
        '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip\n\n',
        '_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip\n'
        "\n"
        "\n"
        "def _aiter_unshuffle_weight(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:\n"
        '    """Inverse of aiter.ops.shuffle.shuffle_weight for its default (non-int4) path."""\n'
        "    BN, IK = layout\n"
        "    BK = IK * 2\n"
        "    K = 16 // x.element_size()\n"
        "    n, k = x.shape[-2], x.shape[-1]\n"
        "    y = x.view(-1, n // BN, k // BK, BK // K, BN, K)\n"
        "    return y.permute(0, 1, 4, 2, 3, 5).contiguous().view(*x.shape)\n"
        "\n",
    ),
    (
        "        expert_id: int,\n    ) -> None:\n        tp_rank = self.moe_tp_rank\n",
        "        expert_id: int,\n"
        "    ) -> None:\n"
        "        # After process_weights_after_loading, aiter MoE weights are tile-shuffled\n"
        "        # per expert; a hot update must land in that layout, not checkpoint layout.\n"
        '        shuffled = getattr(self, "_aiter_moe_weights_shuffled", False) and (\n'
        '            param is getattr(self, "w13_weight", None)\n'
        '            or param is getattr(self, "w2_weight", None)\n'
        "        )\n"
        "        if not shuffled:\n"
        "            return self._weight_loader_impl_canonical(\n"
        "                param, loaded_weight, weight_name, shard_id, expert_id\n"
        "            )\n"
        "\n"
        "        from aiter.ops.shuffle import shuffle_weight\n"
        "\n"
        "        expert_view = param.data[expert_id]\n"
        "        expert_view.copy_(_aiter_unshuffle_weight(expert_view))\n"
        "        try:\n"
        "            self._weight_loader_impl_canonical(\n"
        "                param, loaded_weight, weight_name, shard_id, expert_id\n"
        "            )\n"
        "        finally:\n"
        "            expert_view.copy_(shuffle_weight(expert_view.clone(), (16, 16)))\n"
        "\n"
        "    def _weight_loader_impl_canonical(\n"
        "        self,\n"
        "        param: torch.nn.Parameter,\n"
        "        loaded_weight: torch.Tensor,\n"
        "        weight_name: str,\n"
        "        shard_id: str,\n"
        "        expert_id: int,\n"
        "    ) -> None:\n"
        "        tp_rank = self.moe_tp_rank\n",
    ),
]

UNQUANT_RELATIVE = Path("srt/layers/quantization/unquant.py")
UNQUANT_EXPECTED_SHA256 = (
    "12041329003b0f0bcbe9ec84c80743d4ca053f1d6e443180cacef067b282983b"
)
UNQUANT_PATCHED_SHA256 = (
    "6689c7984195bb3b78e52c65217c6964b7bd374cfedb5831a3cdcd8763e21f2d"
)
UNQUANT_REPLACEMENTS = [
    (
        '                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))\n'
        "            )\n"
        "            torch.cuda.empty_cache()\n"
        "\n",
        '                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))\n'
        "            )\n"
        "            torch.cuda.empty_cache()\n"
        "            # Hot weight updates write checkpoint-layout rows into these buffers;\n"
        "            # FusedMoE.weight_loader reads this flag to keep them in aiter layout.\n"
        "            layer._aiter_moe_weights_shuffled = True\n"
        "\n",
    ),
]


def patched_layer_source(source: bytes) -> bytes:
    """Unshuffle, load and reshuffle an aiter MoE expert in the weight loader."""
    return patch_source(
        source,
        target_relative=LAYER_RELATIVE,
        replacements_by_sha256={LAYER_EXPECTED_SHA256: LAYER_REPLACEMENTS},
        patched_sha256=LAYER_PATCHED_SHA256,
    )


def patched_unquant_source(source: bytes) -> bytes:
    """Mark FusedMoE layers whose weights were aiter-shuffled after loading."""
    return patch_source(
        source,
        target_relative=UNQUANT_RELATIVE,
        replacements_by_sha256={UNQUANT_EXPECTED_SHA256: UNQUANT_REPLACEMENTS},
        patched_sha256=UNQUANT_PATCHED_SHA256,
    )


PATCHES = (
    (LAYER_RELATIVE, patched_layer_source),
    (UNQUANT_RELATIVE, patched_unquant_source),
)


def install(root: Path) -> None:
    install_source_patches(
        [(root / relative, transform) for relative, transform in PATCHES],
        lock_path=root / ".areal-sglang-patches.lock",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="Disposable copy of the sglang package directory"
    )
    root = parser.parse_args().root
    install(root if root is not None else installed_sglang_root())


if __name__ == "__main__":
    main()
