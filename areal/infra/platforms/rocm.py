# SPDX-License-Identifier: Apache-2.0

import gc
import os

import torch

import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("ROCmPlatform")


def _parse_cpu_list(cpulist: str) -> set[int]:
    """Parse a sysfs cpulist string (e.g. "0-95,192-287") into CPU ids."""
    cpus: set[int] = set()
    for chunk in cpulist.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(chunk))
    return cpus


class ROCmPlatform(Platform):
    """AMD GPUs via ROCm.

    ROCm builds of PyTorch expose the CUDA API surface (``torch.cuda``,
    ``dispatch_key="CUDA"``), so this platform mirrors CudaPlatform rather
    than defining a new device type. The differences that matter are the
    device management library (amdsmi instead of pynvml) and the allocator
    env var (``PYTORCH_HIP_ALLOC_CONF``).
    """

    device_name: str = "AMD"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    # ROCm honors CUDA_VISIBLE_DEVICES as an alias for HIP_VISIBLE_DEVICES,
    # and parts of AReaL read CUDA_VISIBLE_DEVICES directly rather than going
    # through device_control_env_var (e.g. awex/colocate_writer.py, which
    # treats it as the only ground truth for relative -> physical GPU ids).
    # Keep the CUDA name so those paths stay correct.
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"
    # RCCL is registered under the "nccl" backend name in torch.distributed.
    communication_backend: str = "nccl"

    def clear_memory(self) -> None:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        # hipBLAS workspaces are managed through the same torch entry point.
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def get_vllm_worker_class(cls):
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
    def set_numa_affinity(cls, local_rank: int) -> None:
        """Bind the current process to CPU cores local to the assigned GPU.

        amdsmi has no equivalent of ``nvmlDeviceSetCpuAffinity``, so resolve
        the GPU's NUMA node and apply that node's CPU list ourselves.
        """

        amdsmi_initialized = False
        try:
            import amdsmi

            # amdsmi enumerates GPUs in its own order, which does NOT match
            # HIP's device order, so handles must not be indexed by rank.
            # Go through the PCI address instead: torch.cuda enumeration
            # already accounts for the visible-device mask, and the BDF is
            # stable across both libraries.
            props = torch.cuda.get_device_properties(local_rank)
            bdf = (
                f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
                f"{props.pci_device_id:02x}.0"
            )

            amdsmi.amdsmi_init()
            amdsmi_initialized = True
            handle = amdsmi.amdsmi_get_processor_handle_from_bdf(bdf)
            numa_node = amdsmi.amdsmi_get_gpu_topo_numa_affinity(handle)
            if numa_node < 0:
                logger.warning("GPU %s reports no NUMA affinity, skipping.", bdf)
                return

            with open(f"/sys/devices/system/node/node{numa_node}/cpulist") as f:
                cpus = _parse_cpu_list(f.read())
            if not cpus:
                logger.warning("NUMA node %s has an empty CPU list.", numa_node)
                return

            os.sched_setaffinity(0, cpus)
            logger.info(
                "Set NUMA affinity for GPU %s (%s, NUMA node %s): bound to %s CPU cores.",
                local_rank,
                bdf,
                numa_node,
                len(os.sched_getaffinity(0)),
            )
        except ImportError:
            logger.warning("amdsmi not available, skipping NUMA affinity setup.")
        except Exception as e:
            logger.warning("Failed to set NUMA affinity for GPU %s: %s", local_rank, e)
        finally:
            if amdsmi_initialized:
                amdsmi.amdsmi_shut_down()

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        }
        return env_vars

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()
