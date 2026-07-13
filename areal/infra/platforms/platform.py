# SPDX-License-Identifier: Apache-2.0

import os

import torch

import areal.utils.logging as logging

logger = logging.getLogger("Platform")


class Platform:
    """
    A unified abstraction for different hardware platforms (e.g., GPU, NPU).
    Design Overview
    ----------------
    1. Device-Agnostic Abstraction
       Hardware platforms differ in how they are registered with PyTorch or Ray.
       This class standardizes platform metadata such as `dispatch_key`, `ray_device_key`,
       and device visibility control variables to simplify cross-platform scheduling.
    2. Lazy Attribute Access
       Dynamically delegates unknown attributes to the `torch.<device_type>` submodule.
       This provides clean access to device-specific PyTorch APIs without redundancy.
    3. Extensible Interface
       Subclasses must implement:
       - `clear_cublas_workspaces`: to release or reuse low-level library workspaces.
       - `get_vllm_worker_class`: to specify the vLLM Ray worker class.
       - `set_allocator_settings`: to configure platform-specific memory allocators.
    """

    # High-level platform name, used for readability and logging.
    # Examples: "NVIDIA", "ASCEND"
    device_name: str

    # Corresponding torch module name
    # Examples: "cuda", "npu"
    device_type: str

    # available dispatch keys:
    # check https://github.com/pytorch/pytorch/blob/313dac6c1ca0fa0cde32477509cce32089f8532a/torchgen/model.py#L134 # noqa
    # use "CPU" as a fallback for platforms not registered in PyTorch
    # Examples: "CUDA", "PrivateUse1"
    dispatch_key: str

    # available ray device keys:
    # https://github.com/ray-project/ray/blob/10ba5adadcc49c60af2c358a33bb943fb491a171/python/ray/_private/ray_constants.py#L438 # noqa
    # empty string means the device does not support ray
    # Examples: "GPU", "NPU"
    ray_device_key: str

    # platform-agnostic way to specify the device control environment variable,
    # .e.g. CUDA_VISIBLE_DEVICES for CUDA.
    # hint: search for "get_visible_accelerator_ids_env_var" in
    # https://github.com/ray-project/ray/tree/master/python/ray/_private/accelerators # noqa
    # Examples: "CUDA_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"
    device_control_env_var: str

    # Optional Ray experimental config
    # Some accelerators require specific flags in Ray start parameters;
    # leave blank if not needed
    # Example: "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str

    # Communication backend for distributed training
    # Examples: "nccl", "hccl"
    communication_backend: str

    def __getattr__(self, key: str):
        """Fallback attribute accessor for device-specific Torch modules.
        This method is called when the requested attribute `key` is not found
        in the current instance. It attempts to retrieve the attribute from
        the corresponding `torch.<device_type>` module (e.g., torch.cuda, torch.xpu).
        If the attribute exists on the device module, it returns it.
        Otherwise, it raises AttributeError.
        Args:
            key (str): The name of the attribute to access.
        Returns:
            Any: The requested attribute from the Torch device module if found.
        Raises:
            AttributeError: If the attribute is not found on the device module.
        """
        device = getattr(torch, self.device_type, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def clear_memory(self) -> None:
        raise NotImplementedError()

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        raise NotImplementedError()

    @classmethod
    def get_vllm_worker_class(cls):
        """Return the custom vLLM WorkerWrapper class used in Ray."""
        raise NotImplementedError()

    @classmethod
    def set_allocator_settings(cls) -> None:
        """Configure memory allocator settings based on the device type."""
        raise NotImplementedError()

    @classmethod
    def set_numa_affinity(cls, local_rank: int) -> None:
        """Bind the current process to CPU cores near the assigned device."""
        return

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        """
        Return custom environment variables specific to the platform.
        Returns:
            dict: A dictionary of environment variable key-value pairs.
        """
        raise NotImplementedError()

    @classmethod
    def update_env_vars_for_visible_devices(
        cls, env_vars: dict, gpu_ranks: list
    ) -> None:
        """
        Update environment variables to control device visibility.
        Args:
            env_vars (dict): Dictionary of current environment variables to modify.
            gpu_ranks (list): List of device IDs to expose to the process.
        Behavior:
            - Sets the platform-specific visibility environment variable.
            - Sets the corresponding Ray experimental flag if needed.
        """
        visible_devices_env_vars = {
            cls.device_control_env_var: ",".join(map(str, gpu_ranks)),
            cls.ray_experimental_noset: "1",
        }
        env_vars.update(visible_devices_env_vars)

    @classmethod
    def get_visible_devices(cls) -> list:
        """
        Return the list of currently visible device IDs.
        Returns:
            list: A list of device ID strings parsed from the visibility environment variable.
        """
        if cls.device_control_env_var is not None:
            return os.environ.get(cls.device_control_env_var, "").split(",")
        return []
