# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging.version import Version


def is_available(pkg_name):
    try:
        return bool(get_version(pkg_name))
    except PackageNotFoundError:
        return False


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.

    :param version1: First version string.
    :param version2: Second version string.
    :return: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
    """
    v1 = Version(version1)
    v2 = Version(version2)
    if v1 < v2:
        return -1
    elif v1 == v2:
        return 0
    else:
        return 1


def is_version_greater_or_equal(package_name: str, target_version: str) -> bool:
    """
    Check if the installed version of a package is greater than or equal to the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is greater than or equal to the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) >= 0


def is_version_less(package_name: str, target_version: str) -> bool:
    """
    Check if the installed version of a package is less than the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is less than the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) < 0


def is_version_equal(package_name: str, target_version: str) -> bool:
    """
    Check if the installed version of a package is equal to the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is equal to the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) == 0
