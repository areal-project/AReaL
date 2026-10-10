# SPDX-License-Identifier: Apache-2.0

"""Complete Qwen4-Exp text-training exports with fixed HF vision assets."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch.distributed as dist

_HF_AUXILIARY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _preserve_hf_auxiliary_files(
    source_path: str, output_path: str
) -> dict[str, list[str]]:
    """Fill missing runtime assets after tokenizer/processor exports take priority."""
    source = Path(source_path)
    output = Path(output_path)
    copied = []
    preserved = []
    for filename in _HF_AUXILIARY_FILES:
        source_file = source / filename
        output_file = output / filename
        if output_file.is_symlink():
            raise ValueError(
                f"Exported HF auxiliary file must not be a symlink: {output_file}"
            )
        if output_file.exists() and not output_file.is_file():
            raise ValueError(
                f"Exported HF auxiliary path must be a regular file: {output_file}"
            )
        if output_file.is_file():
            preserved.append(filename)
        elif source_file.is_file():
            shutil.copy2(source_file, output_file)
            copied.append(filename)
    return {"copied": copied, "preserved_existing": preserved}


def finalize_mcore_bridge_checkpoint(
    source_path: str,
    output_path: str,
    *,
    hf_config: Any,
    language_model_only: bool,
    mtp_enabled: bool,
    cpu_group: dist.ProcessGroup,
    tokenizer: Any | None = None,
    processor: Any | None = None,
) -> dict[str, Any]:
    """Publish fixed weights and metadata on rank zero; broadcast errors to peers.

    All ranks must call after the bridge's tensor-save collectives have returned.
    This cannot recover a failed rank inside the upstream tensor-save collectives.
    """
    result: list[Any] = [None, None]
    if dist.get_rank(group=cpu_group) == 0:
        try:
            config = hf_config
            report = {}
            if hf_config.model_type == "qwen4_exp":
                from mcore_bridge.utils.qwen4_exp_checkpoint import (
                    qwen4_exp_export_config,
                    restore_qwen4_exp_fixed_assets,
                )

                report = restore_qwen4_exp_fixed_assets(
                    source_path,
                    output_path,
                    language_model_only=language_model_only,
                    mtp_enabled=mtp_enabled,
                )
                config = qwen4_exp_export_config(hf_config, mtp_enabled=mtp_enabled)
            config.save_pretrained(output_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_path)
            if processor is not None:
                processor.save_pretrained(output_path)
            report["auxiliary_files"] = _preserve_hf_auxiliary_files(
                source_path, output_path
            )
            result[0] = report
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(
        result,
        src=dist.get_global_rank(cpu_group, 0),
        group=cpu_group,
    )
    if result[1] is not None:
        raise RuntimeError(f"mcore-bridge checkpoint finalization failed: {result[1]}")
    return result[0]
