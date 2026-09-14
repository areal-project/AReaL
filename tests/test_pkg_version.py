# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError
from unittest import mock

from areal.utils import pkg_version


def test_missing_package_comparisons_return_false():
    with mock.patch.object(
        pkg_version, "get_version", side_effect=PackageNotFoundError("missing-pkg")
    ):
        assert pkg_version.is_version_greater_or_equal("missing-pkg", "1.0.0") is False
        assert pkg_version.is_version_less("missing-pkg", "1.0.0") is False
        assert pkg_version.is_version_equal("missing-pkg", "1.0.0") is False


def test_installed_package_comparisons():
    with mock.patch.object(pkg_version, "get_version", return_value="1.2.3"):
        assert pkg_version.is_version_greater_or_equal("pkg", "1.2.3") is True
        assert pkg_version.is_version_greater_or_equal("pkg", "1.2.4") is False
        assert pkg_version.is_version_less("pkg", "1.2.4") is True
        assert pkg_version.is_version_less("pkg", "1.2.3") is False
        assert pkg_version.is_version_equal("pkg", "1.2.3") is True
        assert pkg_version.is_version_equal("pkg", "1.2.4") is False
