# SPDX-License-Identifier: Apache-2.0

import gc

import torch

from .platform import Platform


class XpuPlatform(Platform):
    device_name: str = "Intel XPU"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    ray_device_key: str = "XPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"
    ray_experimental_noset: str = ""
    communication_backend: str = "xccl"

    def clear_memory(self) -> None:
        self.synchronize()
        gc.collect()
        torch.xpu.empty_cache()

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        pass

    @classmethod
    def set_allocator_settings(cls) -> None:
        pass

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        return {}

    @classmethod
    def update_env_vars_for_visible_devices(
        cls, env_vars: dict, gpu_ranks: list
    ) -> None:
        env_vars[cls.device_control_env_var] = ",".join(map(str, gpu_ranks))

    @classmethod
    def synchronize(cls) -> None:
        torch.xpu.synchronize()
