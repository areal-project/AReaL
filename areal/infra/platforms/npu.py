# SPDX-License-Identifier: Apache-2.0

import gc

import torch

import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("NPUPlatform")


class NPUPlatform(Platform):
    device_name: str = "NPU"
    device_type: str = "npu"
    dispatch_key: str = "NPU"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_NPU_VISIBLE_DEVICES"
    communication_backend: str = "hccl"

    @classmethod
    def synchronize(cls) -> None:
        torch.npu.synchronize()

    def clear_memory(self) -> None:
        self.synchronize()
        gc.collect()
        torch.npu.empty_cache()
