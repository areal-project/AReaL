"""The colocate plugin patches SGLang internals, so it pins the versions it knows."""

from unittest import mock

import pytest

from areal.engine.weight_update.awex import v1_sglang_plugin


def test_register_rejects_unverified_sglang_version():
    with mock.patch.object(
        v1_sglang_plugin.pkg_version, "get_version", return_value="0.5.11"
    ):
        with pytest.raises(RuntimeError, match="0.5.11"):
            v1_sglang_plugin.assert_supported_sglang_version()


def test_register_accepts_verified_sglang_versions():
    for version in v1_sglang_plugin.SUPPORTED_SGLANG_VERSIONS:
        with mock.patch.object(
            v1_sglang_plugin.pkg_version, "get_version", return_value=version
        ):
            v1_sglang_plugin.assert_supported_sglang_version()


def test_error_names_the_supported_versions():
    with mock.patch.object(
        v1_sglang_plugin.pkg_version, "get_version", return_value="0.4.0"
    ):
        with pytest.raises(RuntimeError) as excinfo:
            v1_sglang_plugin.assert_supported_sglang_version()
    message = str(excinfo.value)
    for version in v1_sglang_plugin.SUPPORTED_SGLANG_VERSIONS:
        assert version in message
