#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Dynamic Installation Validation Script for AReaL

This script validates that all dependencies listed in pyproject.toml are properly
installed with correct versions and that CUDA extensions are functional.
"""

import sys
from pathlib import Path

from validation_base import BaseInstallationValidator


class DynamicInstallationValidator(BaseInstallationValidator):
    """Validates installation based on pyproject.toml dependencies."""

    def get_validation_title(self) -> str:
        """Get the title for validation output."""
        return "AReaL Installation Validation"


def main():
    """Main entry point."""
    # Find pyproject.toml
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found at {pyproject_path}")
        sys.exit(1)

    validator = DynamicInstallationValidator(pyproject_path)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
