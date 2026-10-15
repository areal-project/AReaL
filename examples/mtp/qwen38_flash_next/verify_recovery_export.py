# SPDX-License-Identifier: Apache-2.0
"""Verify the recovered DCP checkpoint and strict native-MTP HF export."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_digest(path: Path) -> tuple[int, str]:
    with path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = stream.read(header_length)
    return header_length, hashlib.sha256(header).hexdigest()


def _load_tensor(root: Path, index: dict, key: str) -> torch.Tensor:
    shard = root / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _max_diff(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    reference = reference.float()
    actual = actual.float()
    absolute = (reference - actual).abs()
    relative = absolute / reference.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return float(absolute.max()), float(relative.max())


def _verify_frozen_export(
    source_root: Path, export_root: Path, mtp_keys: set[str]
) -> dict:
    """Compare all frozen HF payloads, including ordinary backbone parameters."""
    from examples.mtp.qwen38_flash_next.verify_source_roundtrip import verify

    result = verify(source_root, export_root, excluded_keys=mtp_keys)
    if result["result"] != "PASS":
        raise AssertionError(f"Frozen non-MTP HF export changed: {result}")
    return result


def main() -> None:
    source_root = Path(os.environ["MODEL_PATH"])
    trial_root = Path(os.environ["RECOVERY_TRIAL_ROOT"])
    export_root = Path(os.environ["HF_EXPORT_ROOT"])
    export_indexes = sorted(export_root.rglob("model.safetensors.index.json"))
    if len(export_indexes) != 1:
        raise AssertionError(
            f"expected exactly one HF export index below {trial_root}, got {export_indexes}"
        )
    export_root = export_indexes[0].parent
    source_index = json.loads(
        (source_root / "model.safetensors.index.json").read_text()
    )
    source_config_sha256 = _sha256(source_root / "config.json")
    source_index_sha256 = _sha256(source_root / "model.safetensors.index.json")
    expected_config_sha256 = os.environ["QWEN38_MTP_MODEL_CONFIG_SHA256"]
    expected_index_sha256 = os.environ["QWEN38_MTP_MODEL_INDEX_SHA256"]
    if source_config_sha256 != expected_config_sha256:
        raise AssertionError(
            "source config identity changed: "
            f"{source_config_sha256} != {expected_config_sha256}"
        )
    if source_index_sha256 != expected_index_sha256:
        raise AssertionError(
            "source checkpoint index identity changed: "
            f"{source_index_sha256} != {expected_index_sha256}"
        )
    export_index = json.loads(export_indexes[0].read_text())
    source_keys = set(source_index["weight_map"])
    export_keys = set(export_index["weight_map"])
    missing = sorted(source_keys - export_keys)
    extra = sorted(export_keys - source_keys)
    if missing or extra:
        raise AssertionError(
            f"HF export key mismatch: missing={missing}, extra={extra}"
        )

    mtp_keys = sorted(key for key in source_keys if "mtp" in key.split(".")[:-1])
    ple_keys = sorted(key for key in source_keys if ".ple." in key)
    vision_keys = sorted(key for key in source_keys if key.startswith("model.visual."))
    if len(mtp_keys) != 31:
        raise AssertionError(f"expected 31 native MTP tensors, got {len(mtp_keys)}")

    changed_mtp = []
    mtp_max_abs = 0.0
    mtp_max_rel = 0.0
    for key in mtp_keys:
        source = _load_tensor(source_root, source_index, key)
        exported = _load_tensor(export_root, export_index, key)
        if source.shape != exported.shape or source.dtype != exported.dtype:
            raise AssertionError(
                f"MTP metadata changed for {key}: "
                f"{source.shape}/{source.dtype} != {exported.shape}/{exported.dtype}"
            )
        if not torch.isfinite(exported.float()).all():
            raise AssertionError(f"nonfinite exported MTP tensor: {key}")
        abs_diff, rel_diff = _max_diff(source, exported)
        mtp_max_abs = max(mtp_max_abs, abs_diff)
        mtp_max_rel = max(mtp_max_rel, rel_diff)
        if not torch.equal(source, exported):
            changed_mtp.append(key)
    if not changed_mtp:
        raise AssertionError("HF export contains no MTP update relative to the source")

    frozen_export = _verify_frozen_export(source_root, export_root, set(mtp_keys))
    preserved_ple = len(set(ple_keys) - set(mtp_keys))
    preserved_vision = len(set(vision_keys) - set(mtp_keys))

    shard_manifest = []
    for name in sorted(set(export_index["weight_map"].values())):
        path = export_root / name
        header_length, header_sha256 = _header_digest(path)
        shard_manifest.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "header_length": header_length,
                "header_sha256": header_sha256,
            }
        )
    source_shards = sorted(set(source_index["weight_map"].values()))

    recover_payloads = sorted(trial_root.rglob("recover_checkpoint"))
    if len(recover_payloads) != 1:
        raise AssertionError(
            f"expected exactly one DCP recovery payload, got {recover_payloads}"
        )
    dcp_files = sorted(
        path for path in recover_payloads[0].rglob("*") if path.is_file()
    )
    metadata = [path for path in dcp_files if path.name == ".metadata"]
    if len(metadata) != 1:
        raise AssertionError(f"expected one DCP .metadata, got {metadata}")
    step_infos = sorted(trial_root.rglob("recover_info/step_info.json"))
    if len(step_infos) != 1:
        raise AssertionError(f"expected one recovery step_info, got {step_infos}")
    step_info = json.loads(step_infos[0].read_text())
    if step_info["global_step"] != int(os.environ.get("SAVED_GLOBAL_STEP", "3")):
        raise AssertionError(f"unexpected saved global_step, got {step_info}")

    config = json.loads((export_root / "config.json").read_text())
    result = {
        "source_config_sha256": source_config_sha256,
        "source_index_sha256": source_index_sha256,
        "source_key_count": len(source_keys),
        "source_shard_count": len(source_shards),
        "source_payload_bytes": sum(
            (source_root / name).stat().st_size for name in source_shards
        ),
        "export_key_count": len(export_keys),
        "missing_keys": missing,
        "extra_keys": extra,
        "frozen_non_mtp_export": frozen_export,
        "native_mtp_key_count": len(mtp_keys),
        "changed_mtp_key_count": len(changed_mtp),
        "mtp_max_abs_change_from_source": mtp_max_abs,
        "mtp_max_rel_change_from_source": mtp_max_rel,
        "ple_key_count": len(ple_keys),
        "exactly_preserved_ple_key_count": preserved_ple,
        "vision_key_count": len(vision_keys),
        "exactly_preserved_vision_key_count": preserved_vision,
        "export_shard_count": len(shard_manifest),
        "export_payload_bytes": sum(item["size"] for item in shard_manifest),
        "export_index_sha256": _sha256(export_indexes[0]),
        "export_config_sha256": _sha256(export_root / "config.json"),
        "export_model_type": config.get("model_type"),
        "export_shard_headers": shard_manifest,
        "dcp_file_count": len(dcp_files),
        "dcp_payload_bytes": sum(path.stat().st_size for path in dcp_files),
        "dcp_metadata_sha256": _sha256(metadata[0]),
        "recover_step_info": step_info,
        "recover_step_info_sha256": _sha256(step_infos[0]),
    }
    output = export_root / "qwen38_mtp_export_verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("verify_recovery_export PASS")


if __name__ == "__main__":
    main()
