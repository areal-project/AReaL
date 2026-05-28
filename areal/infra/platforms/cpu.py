# SPDX-License-Identifier: Apache-2.0

from .platform import Platform


class CpuPlatform(Platform):
    device_name: str = "CPU"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "CPU"
    device_control_env_var: str = ""
    ray_experimental_noset: str = ""
    communication_backend: str = "gloo"

    def clear_memory(self) -> None:
        pass

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        pass

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        return {}

    @classmethod
    def synchronize(cls) -> None:
        pass

    @classmethod
    def update_env_vars_for_visible_devices(
        cls, env_vars: dict, gpu_ranks: list
    ) -> None:
        # No-op for CPU platform
        pass

    @classmethod
    def get_visible_devices(cls) -> list:
        # No-devices for CPU platform
        return []

    def memory_allocated(self) -> int:
        """No device memory; return 0."""
        return 0

    def memory_reserved(self) -> int:
        """No device memory; return 0."""
        return 0

    def mem_get_info(self) -> tuple[int, int]:
        """No device memory; return (0, 0)."""
        return (0, 0)

    def empty_cache(self) -> None:
        """No-op."""
        pass
