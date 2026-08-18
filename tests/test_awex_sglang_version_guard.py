"""The colocate plugin patches SGLang internals, so it pins the versions it knows."""

from unittest import mock

import pytest

from areal.engine.awex import sglang_plugin


def test_register_rejects_unverified_sglang_version():
    with mock.patch.object(
        sglang_plugin.pkg_version, "get_version", return_value="0.5.11"
    ):
        with pytest.raises(RuntimeError, match="0.5.11"):
            sglang_plugin.assert_supported_sglang_version()


def test_register_accepts_verified_sglang_versions():
    for version in sglang_plugin.SUPPORTED_SGLANG_VERSIONS:
        with mock.patch.object(
            sglang_plugin.pkg_version, "get_version", return_value=version
        ):
            sglang_plugin.assert_supported_sglang_version()


def test_error_names_the_supported_versions():
    with mock.patch.object(
        sglang_plugin.pkg_version, "get_version", return_value="0.4.0"
    ):
        with pytest.raises(RuntimeError) as excinfo:
            sglang_plugin.assert_supported_sglang_version()
    message = str(excinfo.value)
    for version in sglang_plugin.SUPPORTED_SGLANG_VERSIONS:
        assert version in message
