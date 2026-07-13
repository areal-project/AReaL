# SPDX-License-Identifier: Apache-2.0

"""
Base validation module for AReaL installation validation.

This module provides a base class with common validation logic that can be
extended for different validation scenarios (standard installation, Docker, etc.).
"""

import importlib
import sys
from importlib.metadata import version as get_version
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python 3.10
    except ImportError:
        print("Error: tomllib/tomli not available. Install tomli: pip install tomli")
        sys.exit(1)

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


class BaseInstallationValidator:
    """Base class for validating installation dependencies.

    Note: sglang and vllm are intentionally excluded from CRITICAL_PACKAGES
    because they are mutually exclusive inference backends. Subclasses must
    validate them dynamically based on the environment (e.g.,
    DockerInstallationValidator detects which variant is installed at runtime).
    """

    # Map package names to their import names (when different)
    PACKAGE_IMPORT_MAP = {
        "hydra-core": "hydra",
        "nvidia-cudnn-cu12": "nvidia.cudnn",
        "pillow": "PIL",
        "pyyaml": "yaml",
        "python-dotenv": "dotenv",
        "megatron-core": "megatron.core",
        "PyYAML": "yaml",
        "python-dateutil": "dateutil",
        "python_dateutil": "dateutil",
        "pyzmq": "zmq",
        "nvidia-ml-py": "pynvml",
        "camel-ai": "camel",
        "python-debian": "debian",
        "python_debian": "debian",
        "openai-agents": "agents",
        "tensorboardx": "tensorboardX",
        "megatron-bridge": "megatron.bridge",
    }

    # Map packages to their CUDA sub-modules for deep validation
    # Subclasses can override or extend this
    CUDA_SUBMODULES = {
        "torch": ["torch.cuda"],
        "sglang": ["sgl_kernel", "sgl_kernel.flash_attn"],
        "vllm": ["vllm._C"],
        "flash-attn": ["flash_attn_2_cuda"],
        "megatron-core": [
            "megatron.core.parallel_state",
            "megatron.core.tensor_parallel",
        ],
    }

    # Packages to treat as critical (always fail if missing)
    # Subclasses can override this
    CRITICAL_PACKAGES = {
        "torch",
        "transformers",
        "flash-attn",
        # sglang/vllm are NOT listed here — they are mutually exclusive
        # and should be validated dynamically by subclasses
        "megatron-core",
        "mbridge",
        "megatron-bridge",
        "ray",
        "datasets",
        "hydra-core",
        "omegaconf",
        "wandb",
        "fastapi",
        "uvicorn",
    }

    def __init__(self, pyproject_path: Path | None = None):
        self.pyproject_path = pyproject_path
        self.dependencies = {}
        self.additional_packages = {}  # For packages not in pyproject.toml
        self.results = {}
        self.critical_failures = []
        self.warnings = []

    def parse_pyproject(self):
        """Parse pyproject.toml and extract dependencies."""
        if self.pyproject_path is None:
            print("No pyproject.toml path provided, skipping dependency parsing")
            return

        try:
            with open(self.pyproject_path, "rb") as f:
                data = tomllib.load(f)

            raw_deps = data.get("project", {}).get("dependencies", [])

            for dep in raw_deps:
                # Parse dependency string using packaging.requirements
                # Handles: package, package==version, package>=version, package[extras], etc.
                try:
                    req = Requirement(dep.strip())

                    # Skip dependencies whose environment markers don't match
                    # the current platform (e.g., macOS-only deps in Docker)
                    if req.marker and not req.marker.evaluate():
                        continue

                    # Convert extras set to string format "[extra1,extra2]"
                    extras_str = ""
                    if req.extras:
                        extras_str = f"[{','.join(sorted(req.extras))}]"

                    # Extract operator and version for backward compatibility
                    # If single specifier, extract operator/version; otherwise empty
                    operator = ""
                    version = ""
                    spec_str = str(req.specifier)

                    if req.specifier and len(req.specifier) == 1:
                        spec = list(req.specifier)[0]
                        operator = spec.operator
                        version = spec.version

                    # Store package info
                    self.dependencies[req.name] = {
                        "raw": dep,
                        "extras": extras_str,
                        "operator": operator,
                        "version": version,
                        "spec": spec_str,
                    }

                except InvalidRequirement as e:
                    print(f"Warning: Failed to parse dependency '{dep}': {e}")
                    continue

            print(f"Parsed {len(self.dependencies)} dependencies from pyproject.toml")

        except FileNotFoundError:
            print(f"Error: {self.pyproject_path} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing pyproject.toml: {e}")
            sys.exit(1)

    def _get_optional_dep_versions(self) -> dict[str, str]:
        """Extract version specifiers from [project.optional-dependencies].

        Returns a dict mapping normalized package names to specifier strings
        (e.g. ``{"sglang": "==0.5.9", "megatron-core": "==0.16.0"}``).
        Self-references and marker-mismatched entries are skipped.
        On any error the method returns an empty dict so callers can
        fall back to hardcoded defaults.
        """
        if self.pyproject_path is None:
            return {}
        try:
            with open(self.pyproject_path, "rb") as f:
                data = tomllib.load(f)

            project_name = data.get("project", {}).get("name", "")
            optional_deps = data.get("project", {}).get("optional-dependencies", {})
            versions: dict[str, str] = {}

            for deps in optional_deps.values():
                for dep in deps:
                    try:
                        req = Requirement(dep.strip())
                        if req.name == project_name:
                            continue
                        if req.marker and not req.marker.evaluate():
                            continue
                        spec_str = str(req.specifier)
                        if spec_str and req.name not in versions:
                            versions[req.name] = spec_str
                    except InvalidRequirement:
                        continue

            return versions
        except Exception:
            return {}

    def add_additional_package(
        self, pkg_name: str, version_spec: str = "", required: bool = True
    ):
        """Add a package that's not in pyproject.toml for validation."""
        # Construct requirement string and parse using packaging.requirements
        req_string = f"{pkg_name}{version_spec}"

        try:
            req = Requirement(req_string)

            # Convert extras set to string format "[extra1,extra2]"
            extras_str = ""
            if req.extras:
                extras_str = f"[{','.join(sorted(req.extras))}]"

            # Extract operator and version for backward compatibility
            operator = ""
            version = ""
            spec_str = str(req.specifier)

            if req.specifier and len(req.specifier) == 1:
                spec = list(req.specifier)[0]
                operator = spec.operator
                version = spec.version

            self.additional_packages[pkg_name] = {
                "raw": req_string,
                "extras": extras_str,
                "operator": operator,
                "version": version,
                "spec": spec_str,
                "required": required,
            }

        except InvalidRequirement as e:
            print(f"Warning: Failed to parse additional package '{req_string}': {e}")
            # Fallback to storing as-is for backward compatibility
            self.additional_packages[pkg_name] = {
                "raw": req_string,
                "extras": "",
                "operator": "",
                "version": "",
                "spec": version_spec,
                "required": required,
            }

    def normalize_package_name(self, pkg_name: str) -> str:
        """Normalize package name (handle dash/underscore differences)."""
        # PyPI normalizes package names: replace _ with - and lowercase
        return pkg_name.lower().replace("_", "-")

    def get_installed_version(self, pkg_name: str) -> str | None:
        """Get installed version of a package."""
        # Try both dash and underscore variants
        variants = [
            pkg_name,
            pkg_name.replace("-", "_"),
            pkg_name.replace("_", "-"),
        ]
        if pkg_name.lower() == "fla":
            variants.append("flash-linear-attention")

        for variant in variants:
            try:
                return get_version(variant)
            except Exception:
                continue

        return None

    def check_version(self, pkg_name: str, spec_str: str) -> tuple[bool, str]:
        """
        Check if installed version matches the specification.

        Returns:
            (matches: bool, message: str)
        """
        installed = self.get_installed_version(pkg_name)

        if installed is None:
            return False, "Package not found in installation"

        if not spec_str:
            return True, f"Installed: {installed} (no version constraint)"

        try:
            spec = SpecifierSet(spec_str)
            installed_ver = Version(installed)

            if installed_ver in spec:
                return True, f"Installed: {installed} (matches {spec_str})"
            else:
                return (
                    False,
                    f"Version mismatch: Expected {spec_str}, found {installed}",
                )

        except Exception as e:
            return False, f"Version check error: {e}"

    def _test_import_direct(self, import_name: str) -> tuple[bool, str | None]:
        """Test importing a package directly in the current process.

        Args:
            import_name: The module name to import

        Returns:
            Tuple of (success: bool, error_message: str | None)
        """
        try:
            importlib.import_module(import_name)
            return True, None
        except ImportError as e:
            return False, str(e)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _run_import_tests(
        self, packages: list[tuple[str, str]]
    ) -> dict[str, tuple[bool, str | None]]:
        """Run import tests sequentially in the current process.

        Args:
            packages: List of (pkg_name, import_name) tuples

        Returns:
            Dict mapping pkg_name to (success, error_message) tuples
        """
        results = {}

        for pkg_name, import_name in packages:
            results[pkg_name] = self._test_import_direct(import_name)

        return results

    def test_cuda_submodules(self, pkg_name: str, module):
        """Test CUDA sub-modules for packages with CUDA dependencies."""
        submodules = self.CUDA_SUBMODULES.get(pkg_name, [])

        for submodule_name in submodules:
            try:
                # Special handling for torch.cuda
                if submodule_name == "torch.cuda":
                    if not module.cuda.is_available():
                        raise RuntimeError("CUDA is not available in PyTorch")
                    print(
                        f"  ├─ CUDA devices: {module.cuda.device_count()}, "
                        f"version: {module.version.cuda}"
                    )
                else:
                    # Import sub-module
                    importlib.import_module(submodule_name)
                    print(f"  ├─ {submodule_name} ✓")

            except Exception as e:
                print(f"  ├─ {submodule_name} ✗ ({str(e)})")
                self.warnings.append(
                    f"{pkg_name} CUDA extension ({submodule_name}): {str(e)}"
                )

    def validate_all_dependencies(self):
        """Validate all dependencies from pyproject.toml."""
        print("\n" + "=" * 70)
        print(self.get_validation_title())
        print("=" * 70)

        # Collect all packages to test
        all_packages = []  # List of (pkg_name, import_name, dep_info, required)

        if self.dependencies:
            for pkg_name, dep_info in self.dependencies.items():
                import_name = self.PACKAGE_IMPORT_MAP.get(
                    pkg_name, pkg_name.replace("-", "_")
                )
                normalized_name = self.normalize_package_name(pkg_name)
                is_critical = normalized_name in self.CRITICAL_PACKAGES
                all_packages.append((pkg_name, import_name, dep_info, is_critical))

        if self.additional_packages:
            for pkg_name, dep_info in self.additional_packages.items():
                import_name = self.PACKAGE_IMPORT_MAP.get(
                    pkg_name, pkg_name.replace("-", "_")
                )
                required = dep_info.get("required", True)
                all_packages.append((pkg_name, import_name, dep_info, required))

        # Run all import tests directly in the current process
        print("\nRunning import tests...")
        packages_to_test = [
            (pkg_name, import_name) for pkg_name, import_name, _, _ in all_packages
        ]
        import_results = self._run_import_tests(packages_to_test)

        # Process and display results
        if self.dependencies:
            print("\n=== Critical Dependencies ===")
            critical_count = 0
            for pkg_name, dep_info in sorted(self.dependencies.items()):
                normalized_name = self.normalize_package_name(pkg_name)
                is_critical = normalized_name in self.CRITICAL_PACKAGES

                if is_critical:
                    critical_count += 1
                    self._process_import_result(
                        pkg_name, dep_info, import_results.get(pkg_name), required=True
                    )

            print(
                f"\n=== Other Dependencies ({len(self.dependencies) - critical_count}) ==="
            )
            for pkg_name, dep_info in sorted(self.dependencies.items()):
                normalized_name = self.normalize_package_name(pkg_name)
                is_critical = normalized_name in self.CRITICAL_PACKAGES

                if not is_critical:
                    self._process_import_result(
                        pkg_name, dep_info, import_results.get(pkg_name), required=False
                    )

        # Validate additional packages
        if self.additional_packages:
            print(f"\n=== Additional Packages ({len(self.additional_packages)}) ===")
            for pkg_name, dep_info in sorted(self.additional_packages.items()):
                required = dep_info.get("required", True)
                self._process_import_result(
                    pkg_name, dep_info, import_results.get(pkg_name), required=required
                )

    def _process_import_result(
        self,
        pkg_name: str,
        dep_info: dict,
        import_result: tuple[bool, str | None] | None,
        required: bool = True,
    ) -> bool:
        """Process the result of a parallel import test."""
        import_name = self.PACKAGE_IMPORT_MAP.get(pkg_name, pkg_name.replace("-", "_"))

        # Handle missing result (shouldn't happen, but be safe)
        if import_result is None:
            import_result = (False, "Import test not run")

        import_ok, import_error = import_result

        if not import_ok:
            self.results[pkg_name] = {"status": "IMPORT_FAILED", "error": import_error}
            if required:
                self.critical_failures.append(
                    f"{pkg_name}: Import failed - {import_error}"
                )
                print(f"✗ {pkg_name} (IMPORT FAILED): {import_error}")
            else:
                self.warnings.append(f"{pkg_name}: Import failed - {import_error}")
                print(f"⚠ {pkg_name} (IMPORT FAILED): {import_error}")
            return False

        # Check version
        version_ok, version_msg = self.check_version(pkg_name, dep_info.get("spec", ""))

        if not version_ok:
            self.results[pkg_name] = {
                "status": "VERSION_MISMATCH",
                "error": version_msg,
            }
            if required:
                self.critical_failures.append(f"{pkg_name}: {version_msg}")
                print(f"✗ {pkg_name} (VERSION MISMATCH): {version_msg}")
            else:
                self.warnings.append(f"{pkg_name}: {version_msg}")
                print(f"⚠ {pkg_name} (VERSION MISMATCH): {version_msg}")
            return False

        # Test CUDA sub-modules if applicable (still use in-process for these)
        if pkg_name in self.CUDA_SUBMODULES:
            try:
                module = importlib.import_module(import_name)
                self.test_cuda_submodules(pkg_name, module)
            except Exception:
                pass  # CUDA submodule errors are already handled as warnings

        self.results[pkg_name] = {"status": "SUCCESS", "error": None}
        print(f"✓ {pkg_name} - {version_msg}")
        return True

    def test_cuda_functionality(self):
        """Run basic CUDA functionality tests. Can be overridden by subclasses."""
        print("\n=== CUDA Functionality Tests ===")

        try:
            import torch

            if not torch.cuda.is_available():
                print("⚠ CUDA not available - skipping CUDA tests")
                return

            # Test basic CUDA operations
            try:
                device = torch.device("cuda:0")
                x = torch.randn(10, device=device)
                y = torch.randn(10, device=device)
                _ = x + y
                print("✓ Basic CUDA tensor operations")
            except Exception as e:
                print(f"✗ Basic CUDA operations failed: {e}")

            # Test flash attention if available
            try:
                from flash_attn import flash_attn_func

                batch_size, seq_len, num_heads, head_dim = 1, 32, 4, 64
                q = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                k = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                v = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                _ = flash_attn_func(q, k, v)
                print("✓ Flash attention CUDA operations")
            except Exception as e:
                print(f"⚠ Flash attention test failed: {e}")

        except ImportError:
            print("⚠ PyTorch not available - skipping CUDA tests")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        total_tests = len(self.results)
        successful_tests = sum(
            1 for r in self.results.values() if r["status"] == "SUCCESS"
        )
        failed_tests = total_tests - successful_tests

        print(f"Total packages tested: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")

        if self.critical_failures:
            print(f"\n🚨 CRITICAL FAILURES ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"  - {failure}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        # Determine overall result
        if self.critical_failures:
            print("\n❌ INSTALLATION VALIDATION FAILED")
            print("Please fix the critical failures above and ensure all required")
            print("dependencies are properly installed.")
            return False
        else:
            print("\n✅ INSTALLATION VALIDATION PASSED")
            if self.warnings:
                print("Note: Some warnings were reported but core functionality")
                print("should not be affected.")
            return True

    def get_validation_title(self) -> str:
        """Get the title for validation output. Can be overridden by subclasses."""
        return "AReaL Installation Validation"

    def run(self):
        """Run the complete validation process."""
        self.parse_pyproject()
        self.validate_all_dependencies()
        self.test_cuda_functionality()
        success = self.print_summary()
        return success
