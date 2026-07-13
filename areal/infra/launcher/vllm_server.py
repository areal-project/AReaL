# SPDX-License-Identifier: Apache-2.0

import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import requests

from areal.api.alloc_mode import _AllocationMode
from areal.api.cli_args import (
    ClusterSpecConfig,
    InferenceEngineConfig,
    NameResolveConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.platforms import current_platform
from areal.infra.utils.launcher import TRITON_CACHE_PATH, get_scheduling_spec
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports, format_hostport, gethostip

logger = logging.getLogger("VLLMWrapper")


def launch_server_cmd(
    command: list[str], custom_env: dict[str, str] | None = None
) -> subprocess.Popen:
    """
    Launch inference server in a new process and return its process handle.
    """
    # Replace newline continuations and split the command string.
    logger.info(f"Launch command: {' '.join(command)}")
    _env = os.environ.copy()
    # To avoid DirectoryNotEmpty error caused by triton
    triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
    unique_triton_cache_path = os.path.join(triton_cache_path, str(uuid.uuid4()))
    _env["TRITON_CACHE_PATH"] = unique_triton_cache_path
    # To avoid vllm compile cache conflict
    vllm_cache_path = _env.get("VLLM_CACHE_ROOT")
    if vllm_cache_path:
        _env["VLLM_CACHE_ROOT"] = os.path.join(vllm_cache_path, str(uuid.uuid4()))
    _env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    if custom_env is not None:
        _env.update(custom_env)

    return subprocess.Popen(
        command,
        env=_env,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )


def wait_for_server(base_url: str, timeout: int | None = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)
                break

            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)


class vLLMServerWrapper:
    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        vllm_config: vLLMConfig,
        allocation_mode: _AllocationMode,
        n_gpus_per_node: int,
        cpu_per_gpu: int | None = None,
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.config = vllm_config
        self.allocation_mode = allocation_mode
        self.server_processes = []
        self.n_gpus_per_node = n_gpus_per_node
        self.cpu_per_gpu = cpu_per_gpu
        self._shutdown_requested = False
        self._is_shutting_down = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals by cleaning up all server processes."""
        if self._is_shutting_down:
            logger.warning("Shutdown already in progress, ignoring redundant signal.")
            return
        self._is_shutting_down = True

        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_requested = True
        self._cleanup_all_servers()
        sys.exit(0)

    def _cleanup_all_servers(self):
        """Clean up all server processes and their children."""
        # To prevent race conditions from re-entrant calls (e.g., from a signal handler),
        # we first copy the list of processes and then clear the original list.
        processes_to_clean = list(self.server_processes)
        self.server_processes.clear()

        if not processes_to_clean:
            logger.info("No server processes to clean up")
            return

        logger.info(f"Cleaning up {len(processes_to_clean)} vLLM server processes")
        for i, process in enumerate(processes_to_clean):
            if process.poll() is None:  # Process is still running
                logger.info(f"Terminating vLLM server process {i} (PID: {process.pid})")
                kill_process_tree(process.pid, timeout=10, graceful=True)
            else:
                logger.info(f"vLLM server process {i} already terminated")

        logger.info("All vLLM server processes cleaned up")

    def run(self):
        gpus_per_server = self.allocation_mode.gen_instance_size
        if gpus_per_server > self.n_gpus_per_node:
            raise NotImplementedError("Cross-node vllm is not supported")

        n_servers_per_node = max(1, self.n_gpus_per_node // gpus_per_server)
        device_control_env_var = current_platform.device_control_env_var
        if device_control_env_var in os.environ:
            visible = os.getenv(device_control_env_var).split(",")
            n_visible_devices = len(visible)
            n_servers_per_proc = max(1, n_visible_devices // gpus_per_server)
            # Use modulo to ensure server_idx_offset is node-local (0 to n_servers_per_node-1)
            # This prevents port overflow when running multiple nodes
            server_idx_offset = (
                min(list(map(int, visible))) // gpus_per_server
            ) % n_servers_per_node
        else:
            visible = [str(i) for i in range(self.n_gpus_per_node)]
            n_servers_per_proc = n_servers_per_node
            server_idx_offset = 0

        # Separate ports used by each server in the same node
        # ports range (10000, 32767)
        ports_per_server = 22767 // n_servers_per_node
        launch_server_args = []
        server_addresses = []
        base_random_seed = self.config.seed
        for j, server_local_idx in enumerate(
            range(server_idx_offset, server_idx_offset + n_servers_per_proc)
        ):
            port_range = (
                server_local_idx * ports_per_server + 10000,
                (server_local_idx + 1) * ports_per_server + 10000,
            )
            server_port, dist_init_port = find_free_ports(2, port_range)

            dist_init_addr = f"localhost:{dist_init_port}"
            host_ip = gethostip()

            custom_env = {
                device_control_env_var: ",".join(
                    visible[j * gpus_per_server : (j + 1) * gpus_per_server]
                ),
            }
            config = deepcopy(self.config)
            config.seed = base_random_seed + server_local_idx
            cmd = vLLMConfig.build_cmd(
                config,
                tp_size=self.allocation_mode.gen.tp_size,
                pp_size=self.allocation_mode.gen.pp_size,
                host=host_ip,
                port=server_port,
                dist_init_addr=dist_init_addr,
            )
            launch_server_args.append((cmd, host_ip, server_port, custom_env))
            server_addresses.append(f"http://{format_hostport(host_ip, server_port)}")

        try:
            with ThreadPoolExecutor(max_workers=n_servers_per_proc) as executor:
                server_iterator = executor.map(
                    lambda args: self.launch_one_server(*args), launch_server_args
                )
                # Consume iterator incrementally to allow cleanup of partially started servers
                for process in server_iterator:
                    self.server_processes.append(process)

            while not self._shutdown_requested:
                all_alive = True
                for i, process in enumerate(self.server_processes):
                    return_code = process.poll()
                    if return_code is not None:
                        logger.info(
                            f"vllm server {server_addresses[i]} exits, "
                            f"returncode={return_code}"
                        )
                        all_alive = False
                        break

                if not all_alive:
                    logger.warning(
                        "One or more vLLM servers died, cleaning up all servers"
                    )
                    self._cleanup_all_servers()
                    sys.exit(1)

                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in vLLM server wrapper: {e}")
            self._cleanup_all_servers()
            raise

    def launch_one_server(
        self,
        cmd: str,
        host_ip: str,
        server_port: int,
        custom_env: dict[str, str] | None = None,
    ):
        server_process = launch_server_cmd(cmd, custom_env=custom_env)
        wait_for_server(f"http://{format_hostport(host_ip, server_port)}")
        name = names.gen_servers(self.experiment_name, self.trial_name)
        name_resolve.add_subentry(name, f"{host_ip}:{server_port}")
        logger.info(
            f"vllm server launched at: http://{format_hostport(host_ip, server_port)}"
        )
        return server_process


def launch_vllm_server(argv):
    config, _ = parse_cli_args(argv)
    config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    config.cluster.name_resolve = to_structured_cfg(
        config.cluster.name_resolve, NameResolveConfig
    )
    config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)
    name_resolve.reconfigure(config.cluster.name_resolve)

    warnings.warn(
        "SPMD launchers use the deprecated _AllocationMode parser which will be removed. "
        "Bare dimension strings (e.g., 'd4t2') are NO LONGER ACCEPTED. "
        "All allocation strings must include an explicit backend prefix "
        "(e.g., 'fsdp:d4', 'sglang:d4t2'). "
        "Migrate to single-controller mode (scheduler.type=local) with per-engine 'backend' "
        "fields (e.g., actor.backend='fsdp:d4'). See docs/en/reference/alloc_mode.md.",
        FutureWarning,
        stacklevel=2,
    )
    allocation_mode = config.allocation_mode
    allocation_mode = _AllocationMode.from_str(allocation_mode)
    assert allocation_mode.gen_backend == "vllm"

    # Get CPU per GPU from rollout scheduling spec
    rollout_spec = get_scheduling_spec(config.rollout)

    vllm_server = vLLMServerWrapper(
        config.experiment_name,
        config.trial_name,
        config.vllm,
        allocation_mode,
        n_gpus_per_node=config.cluster.n_gpus_per_node,
        cpu_per_gpu=rollout_spec.cpu,
    )
    vllm_server.run()


def main(argv):
    try:
        launch_vllm_server(argv)
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        kill_process_tree(graceful=True)


if __name__ == "__main__":
    main(sys.argv[1:])
