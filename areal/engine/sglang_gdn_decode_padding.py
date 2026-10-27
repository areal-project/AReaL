# SPDX-License-Identifier: Apache-2.0
"""Opt-in backport of https://github.com/sgl-project/sglang/pull/44014.

SM90 decode must pass CUDA Graph padding (-1) to FlashInfer's indexed state
pool API. Python gather/scatter instead aliases the last allocatable slot.
Padded lanes can corrupt live or cached recurrent state and affect subsequent
generation.

This patch is not applied automatically. It must be run explicitly in the
SGLang interpreter before starting SGLang servers::

    python -m areal.engine.sglang_gdn_decode_padding

Requires a separately provisioned FlashInfer >= 0.6.18 runtime. The default
AReaL dependency pins are unchanged; their FlashInfer 0.6.7.post3 is rejected.
Validated source profiles: SGLang 0.5.10.post1, 0.5.20, and the upstream PR
base module. Unknown and locally modified sources are rejected before writing.
Use ``--target PATH`` to patch a disposable source copy without importing or
checking the local GPU dependencies. Repeated application is write-free.
"""

import argparse
import hashlib
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from areal.engine._sglang_source_patch import (
    install_source_patches,
    installed_sglang_root,
    patch_source,
)

TARGET_RELATIVE = Path("srt/layers/attention/linear/kernels/gdn_flashinfer.py")
MIN_FLASHINFER_VERSION = (0, 6, 18)

PINNED_OLD = """\
        else:
            # TODO: Once FlashInfer PR#2521 is merged for SM90, gather/scatter
            # will no longer be needed here.
            state_batch = ssm_states[cache_indices]
            output_fi, new_state = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=state_batch,
                A_log=A_log.detach(),
                a=a_fi,
                dt_bias=dt_bias.detach(),
                b=b_fi,
                scale=None,
                output=None,
                use_qk_l2norm=True,
            )
            ssm_states[cache_indices] = new_state
"""

PINNED_NEW = """\
        else:
            # Keep -1 as padding instead of indexing the last live pool slot.
            cache_indices_fi = cache_indices.to(dtype=torch.int32)
            output_fi, _ = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=None,
                A_log=A_log.detach(),
                a=a_fi,
                dt_bias=dt_bias.detach(),
                b=b_fi,
                use_qk_l2norm=True,
                initial_state=ssm_states,
                initial_state_indices=cache_indices_fi,
            )
"""

DYNAMIC_OLD = """\
        else:
            # TODO: Once FlashInfer PR#2521 is merged for SM90, gather/scatter
            # will no longer be needed here.
            state_batch = ssm_states[cache_indices]
            output_fi, new_state = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=state_batch,
                A_log=A_log_fi,
                a=a_fi,
                dt_bias=dt_bias_fi,
                b=b_fi,
                scale=None,
                output=None,
                use_qk_l2norm=True,
            )
            ssm_states[cache_indices] = new_state
"""

DYNAMIC_NEW = """\
        else:
            # Keep -1 as padding instead of indexing the last live pool slot.
            cache_indices_fi = cache_indices.to(dtype=torch.int32)
            cache_indices_fi = self._prepare_dynamic_input(
                "decode_cache_indices", cache_indices_fi
            )
            output_fi, _ = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=None,
                A_log=A_log_fi,
                a=a_fi,
                dt_bias=dt_bias_fi,
                b=b_fi,
                use_qk_l2norm=True,
                initial_state=ssm_states,
                initial_state_indices=cache_indices_fi,
            )
"""

_PATCHES = {
    "9534205439e1408129e9bfb38a2e2b963d030c08d287a493517531a5d2e6b749": (
        PINNED_OLD,
        PINNED_NEW,
        "e2dbb29f33f6a8d65633ed46fcd650e0e854ed142d925e8b8117f14e0fe232af",
    ),
    "6db787654af7953b5572520b9b1b048f8939e3587e8fdb287e82bcce4d168b02": (
        DYNAMIC_OLD,
        DYNAMIC_NEW,
        "57739cd4105ff1926f54ee9e807cf60785a6b639389a0e8b3141b137b1b8c217",
    ),
    "2a3092cef9ac5493ae2a52bb3dd129c729df44782bc27886935efac842687e02": (
        DYNAMIC_OLD,
        DYNAMIC_NEW,
        "e08981fcda1101208965f1ba4c805482457ee00d33bafe07bfdae4d6a36054f0",
    ),
}


def patched_source(source: bytes) -> bytes:
    """Apply the SM90-only backport to a known source, preserving other paths."""
    digest = hashlib.sha256(source).hexdigest()
    if digest in {profile[2] for profile in _PATCHES.values()}:
        return source
    if digest not in _PATCHES:
        raise RuntimeError(
            "Unknown SGLang gdn_flashinfer.py source; refusing to modify"
        )
    old, new, patched_digest = _PATCHES[digest]
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={digest: [(old, new)]},
        patched_sha256=patched_digest,
    )


def validate_flashinfer_version() -> None:
    """Reject old or unrecognized runtimes before changing installed source."""
    try:
        installed = version("flashinfer-python")
    except PackageNotFoundError as exc:
        raise RuntimeError("FlashInfer >= 0.6.18 is required for this patch") from exc
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?", installed)
    if match is None or tuple(map(int, match.groups())) < MIN_FLASHINFER_VERSION:
        raise RuntimeError(
            f"FlashInfer >= 0.6.18 is required for this patch; found {installed}. "
            "Use a separately provisioned runtime before applying this backport."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="Disposable source copy")
    target = parser.parse_args().target
    if target is None:
        validate_flashinfer_version()
        root = installed_sglang_root()
        target = root / TARGET_RELATIVE
    else:
        root = target.parent
    install_source_patches(
        [(target, patched_source)], lock_path=root / ".areal-sglang-patches.lock"
    )


if __name__ == "__main__":
    main()
