# SPDX-License-Identifier: Apache-2.0

import gc

import torch

import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("UnknownPlatform")


class UnknownPlatform(Platform):
    device_name: str = "UNKNOWN"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    def clear_memory(self) -> None:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def get_vllm_worker_class(clas):
        try:
            from vllm.v1.worker.gpu_worker import Worker

            logger.info("Successfully imported vLLM V1 Worker.")
            return Worker
        except ImportError:
            pass

        try:
            from vllm.worker.worker import Worker

            logger.info("Successfully imported vLLM V0 Worker.")
            return Worker
        except ImportError as e:
            logger.error(
                "Failed to import vLLM Worker. "
                "Make sure vLLM is installed correctly: %s",
                e,
            )
            raise RuntimeError(
                "vLLM is not installed or not properly configured."
            ) from e

    @classmethod
    def set_allocator_settings(cls) -> None:
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
        }
        return env_vars
