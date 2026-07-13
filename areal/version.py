# SPDX-License-Identifier: Apache-2.0

import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_package_version() -> str:
    """Get version from installed package metadata (single source of truth)."""
    try:
        return version("areal")
    except PackageNotFoundError:
        # Package not installed, fallback for development
        return "0.0.0.dev"


class VersionInfo:
    def __init__(self):
        self.__version__ = _get_package_version()
        self.__branch__ = ""
        self.__commit__ = ""
        self.__is_dirty__ = False
        self._populate_version_info()

    @property
    def version(self) -> str:
        return self.__version__

    @property
    def branch(self) -> str:
        return self.__branch__

    @property
    def commit(self) -> str:
        return self.__commit__

    @property
    def is_dirty(self) -> bool:
        return self.__is_dirty__

    @property
    def full_version(self) -> str:
        version = self.version
        if self.commit != "":
            version = f"{self.version}-{self.commit}"
        if self.is_dirty:
            version = f"{version}-dirty"
        return version

    @property
    def full_version_with_dirty_description(self) -> str:
        version = self.full_version
        if self.is_dirty:
            version = (
                f"{version} ('-dirty' means there are uncommitted code changes in git)"
            )
        return version

    @lru_cache(maxsize=1)
    def _populate_version_info(self):
        try:
            self.__branch__ = (
                subprocess.check_output(
                    ["git", "branch", "--show-current"],
                    stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).parent,
                )
                .decode("utf-8")
                .strip()
            )
            self.__commit__ = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).parent,
                )
                .decode("utf-8")
                .strip()
            )
            git_diff_result = subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            )
            self.__is_dirty__ = git_diff_result.returncode != 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass


version_info = VersionInfo()

# for backward compatibility
__version__ = version_info.version
__branch__ = version_info.branch
__commit__ = version_info.commit
__is_dirty__ = version_info.is_dirty
