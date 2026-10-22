# SPDX-License-Identifier: Apache-2.0

import hashlib
import sys
from types import SimpleNamespace

import pytest

from areal.engine import sglang_gdn_decode_padding as patch


@pytest.fixture(params=["PINNED", "DYNAMIC"])
def source_profile(request, monkeypatch):
    """Exercise both source call sites without optional GPU imports."""
    old = getattr(patch, request.param + "_OLD")
    new = getattr(patch, request.param + "_NEW")
    source = (
        "class Kernel:\n"
        "    def _prepare_dynamic_input(self, name, value):\n"
        "        return value\n"
        "    def decode(self, cache_indices, ssm_states):\n"
        "        query_fi = key_fi = value_fi = a_fi = b_fi = None\n"
        "        A_log_fi = dt_bias_fi = A_log = dt_bias = scalar\n"
        "        if self.use_state_pool:\n"
        "            return 'unchanged-blackwell-path'\n"
        + old
        + "\n        return output_fi\n"
    ).encode()
    expected = source.replace(old.encode(), new.encode())
    monkeypatch.setattr(
        patch,
        "_PATCHES",
        {
            hashlib.sha256(source).hexdigest(): (
                old,
                new,
                hashlib.sha256(expected).hexdigest(),
            )
        },
    )
    return source, expected


def test_padding_reaches_indexed_kernel_without_python_pool_access(source_profile):
    """The old call site touches the pool with -1; the backport must not."""

    class Pool:
        def __getitem__(self, index):
            raise AssertionError("Python pool access would alias the final live slot")

        def __setitem__(self, index, value):
            raise AssertionError("Python scatter would overwrite the final live slot")

    class Indices:
        def __init__(self, values, dtype):
            self.values = values
            self.dtype = dtype

        def to(self, *, dtype):
            return Indices(self.values, dtype)

    scalar = SimpleNamespace(detach=lambda: scalar)
    source, expected = source_profile
    namespace = {"torch": SimpleNamespace(int32="int32"), "scalar": scalar}
    exec(compile(source, "unpatched.py", "exec"), namespace)
    kernel = namespace["Kernel"]()
    kernel.use_state_pool = False
    pool = Pool()
    indices = Indices([1, -1, 3, -1], "int64")
    with pytest.raises(AssertionError, match="final live slot"):
        kernel.decode(indices, pool)

    patched = patch.patched_source(source)
    assert patched == expected
    assert patch.patched_source(patched) == patched
    exec(compile(patched, "patched.py", "exec"), namespace)
    kernel = namespace["Kernel"]()
    kernel.use_state_pool = False
    calls = []

    def decode(**kwargs):
        assert kwargs["state"] is None
        assert kwargs["initial_state"] is pool
        assert kwargs["initial_state_indices"].values == indices.values
        assert kwargs["initial_state_indices"].dtype == "int32"
        calls.append(kwargs)
        return "output", None

    kernel._decode_fn = decode
    assert kernel.decode(indices, pool) == "output"
    assert len(calls) == 1
    kernel.use_state_pool = True
    assert kernel.decode(indices, pool) == "unchanged-blackwell-path"
    assert len(calls) == 1


def test_target_copy_is_idempotent_without_gpu_dependencies(
    source_profile, tmp_path, monkeypatch
):
    """Offline application keeps file permissions and avoids a second write."""
    source, expected = source_profile
    target = tmp_path / "gdn_flashinfer.py"
    target.write_bytes(source)
    target.chmod(0o640)
    monkeypatch.setattr(sys, "argv", ["patch", "--target", str(target)])
    monkeypatch.setattr(
        patch,
        "validate_flashinfer_version",
        lambda: pytest.fail("GPU dependency access"),
    )
    patch.main()
    assert target.read_bytes() == expected
    assert target.stat().st_mode & 0o777 == 0o640
    modified = target.stat().st_mtime_ns
    patch.main()
    assert target.stat().st_mtime_ns == modified


def test_unknown_source_is_not_modified(tmp_path, monkeypatch):
    """A new or locally modified SGLang module needs explicit validation."""
    target = tmp_path / "gdn_flashinfer.py"
    target.write_bytes(b"# unsupported source\n")
    monkeypatch.setattr(sys, "argv", ["patch", "--target", str(target)])
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.main()
    assert target.read_bytes() == b"# unsupported source\n"
    assert not (tmp_path / ".areal-sglang-patches.lock").exists()


@pytest.mark.parametrize("installed", ["0.6.7.post3", "0.6.17", "0.6.18rc1", "dev"])
def test_unsupported_flashinfer_is_rejected_before_source_access(
    monkeypatch, installed
):
    """The default older dependency pins cannot accidentally receive this patch."""
    monkeypatch.setattr(patch, "version", lambda _: installed)
    monkeypatch.setattr(
        patch, "installed_sglang_root", lambda: pytest.fail("source accessed")
    )
    monkeypatch.setattr(sys, "argv", ["patch"])
    with pytest.raises(RuntimeError, match="FlashInfer >= 0.6.18"):
        patch.main()


@pytest.mark.parametrize("installed", ["0.6.18", "0.6.18.post1", "0.7.1"])
def test_supported_flashinfer_passes_version_guard(monkeypatch, installed):
    monkeypatch.setattr(patch, "version", lambda _: installed)
    patch.validate_flashinfer_version()


def test_missing_flashinfer_is_reported(monkeypatch):
    def missing(_):
        raise patch.PackageNotFoundError("flashinfer-python")

    monkeypatch.setattr(patch, "version", missing)
    with pytest.raises(RuntimeError, match="FlashInfer >= 0.6.18"):
        patch.validate_flashinfer_version()
