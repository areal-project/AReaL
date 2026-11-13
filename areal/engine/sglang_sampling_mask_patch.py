# SPDX-License-Identifier: Apache-2.0
"""Fingerprint-checked integration for SGLang 0.5.10.post1."""

from functools import partial
from pathlib import Path

from areal.engine._sglang_source_patch import (
    install_source_patches,
    installed_sglang_root,
    patch_source,
)

_DETERMINISTIC_OLD = '            # Check sampling backend\n            self.sampling_backend = "pytorch"\n            logger.warning(\n                "Sampling backend is set to pytorch for deterministic inference."\n            )'
_DETERMINISTIC_NEW = '            # Keep AReaL\'s restricted policy and its seed-aware sampler.\n            if self.sampling_backend != "areal_sampling_mask":\n                self.sampling_backend = "pytorch"\n                logger.warning(\n                    "Sampling backend is set to pytorch for deterministic inference."\n                )'

_PATCHES = (
    (
        "srt/layers/sampler.py",
        "476be765ba216e8da723e43a1dc0f0038426f68cc7ac2001bc3f47db69521b08",
        "cb4312612c22c3bb94255e62ef8e65f3f0de8c60671b2d98e58f66c4cf8409fb",
        [
            (
                ' the processor to the logits\n        logits[batch_mask] = processor(\n            logits[batch_mask],\n            [sampling_batch_info.custom_params[i] for i in batch_indices],\n        )\n\n        logger.debug(\n            f"Custom logit processor {processor.__class__.__name__} is applied."\n        )\n',
                ' the processor to the logits\n        logits[batch_mask] = processor(\n            logits[batch_mask],\n            [sampling_batch_info.custom_params[i] for i in batch_indices],\n        )\n\n        logger.debug(\n            f"Custom logit processor {processor.__class__.__name__} is applied."\n        )\n\n# AReaL frozen sampling supports: registered in every multiprocessing child.\nimport os as _areal_mask_os\nif _areal_mask_os.environ.get("AREAL_SAMPLING_MASK_TRANSFER") == "1":\n    from areal.engine.sglang_sampling_mask import register_backend as _areal_mask_register\n    _areal_mask_register()\n',
            )
        ],
    ),
    (
        "srt/managers/tokenizer_manager.py",
        "5645ffcd9c61bd1d8c4c4062ebe41a6b48ae8e07891c56e6b2b7dcfb8d16d834",
        "9b09364ae6ab2c1f3321787110e3a6ce6e8345667f02848eb49e66f0ba2b188c",
        [
            (
                "                for k, v in recv_obj.customized_info.items():\n                    meta_info[k] = v[i]",
                "                from areal.engine.sglang_sampling_mask import accumulate_reference_metadata\n                for k, v in recv_obj.customized_info.items():\n                    accumulate_reference_metadata(state, meta_info, k, v[i])",
            )
        ],
    ),
    (
        "srt/server_args.py",
        "921201ad646844722db62e7672b1a8a119d93269d7b39d84140bc2a470035f1d",
        "fe511cce9a4952489e2d2240ecc8b52218b991fc81b40c2bf77384e610b8af00",
        [
            (
                'SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}',
                'SAMPLING_BACKEND_CHOICES = {"flashinfer", "pytorch", "ascend"}\nif os.environ.get("AREAL_SAMPLING_MASK_TRANSFER") == "1":\n    SAMPLING_BACKEND_CHOICES.add("areal_sampling_mask")',
            ),
            (_DETERMINISTIC_OLD, _DETERMINISTIC_NEW),
        ],
    ),
)


def install() -> None:
    root = installed_sglang_root()
    targets = [
        (
            root / relative,
            partial(
                patch_source,
                target_relative=Path(relative),
                replacements_by_sha256={
                    before: replacements,
                    **(
                        {
                            "b8d4b1ae172ea814567025db80f3eacdc4f71af935a894b7db31026ed8e94352": [
                                (_DETERMINISTIC_OLD, _DETERMINISTIC_NEW)
                            ]
                        }
                        if relative == "srt/server_args.py"
                        else {}
                    ),
                },
                patched_sha256=after,
            ),
        )
        for relative, before, after, replacements in _PATCHES
    ]
    install_source_patches(targets, lock_path=root / ".areal-sglang-patches.lock")
