# SPDX-License-Identifier: Apache-2.0

import os
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
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.infra.platforms import current_platform
from areal.infra.utils.launcher import (
    TRITON_CACHE_PATH,
    get_scheduling_spec,
)
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports, format_hostport, gethostip

logger = logging.getLogger("SGLangWrapper")


def launch_server_cmd(
    command: list[str], custom_env: dict[str, str] | None = None
) -> subprocess.Popen:
    """
    Launch inference server in a new process and return its process handle.

    Args:
        command: The command to execute.
        custom_env: Custom environment variables to set for the subprocess.
    """
    # Replace newline continuations and split the command string.
    logger.info(f"Launch command: {' '.join(command)}")
    _env = os.environ.copy()
    # To avoid DirectoryNotEmpty error caused by triton
    triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
    unique_triton_cache_path = os.path.join(triton_cache_path, str(uuid.uuid4()))
    _env["TRITON_CACHE_PATH"] = unique_triton_cache_path

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


class SGLangServerWrapper:
    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        sglang_config: SGLangConfig,
        allocation_mode: _AllocationMode,
        n_gpus_per_node: int,
        cpu_per_gpu: int | None = None,
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.config = sglang_config
        self.allocation_mode = allocation_mode
        self.server_processes = []  # List to store multiple server processes
        self.n_gpus_per_node = n_gpus_per_node
        self.cpu_per_gpu = cpu_per_gpu

    def _monitor_server_processes(self, server_addresses):
        """Monitor server processes and exit if any dies."""
        while True:
            all_alive = True
            for i, process in enumerate(self.server_processes):
                return_code = process.poll()
                if return_code is not None:
                    logger.info(
                        f"SGLang server {server_addresses[i]} exits, returncode={return_code}"
                    )
                    all_alive = False
                    break

            if not all_alive:
                sys.exit(1)

            time.sleep(1)

    def run(self):
        gpus_per_server = self.allocation_mode.gen_instance_size
        cross_nodes = False
        if gpus_per_server > self.n_gpus_per_node:
            if gpus_per_server % self.n_gpus_per_node != 0:
                raise ValueError(
                    "Cross-nodes SGLang only supports utilizing all gpus in one node"
                )
            cross_nodes = True
            node_rank = int(os.environ["AREAL_SGLANG_MULTI_NODE_RANK"])
            master_addr = os.environ["AREAL_SGLANG_MULTI_NODE_MASTER_ADDR"]
            master_port = int(os.environ["AREAL_SGLANG_MULTI_NODE_MASTER_PORT"])
        else:
            node_rank = 0
            master_addr = None
            master_port = None

        n_servers_per_node = max(1, self.n_gpus_per_node // gpus_per_server)
        n_nodes_per_server = max(1, gpus_per_server // self.n_gpus_per_node)

        if current_platform.device_control_env_var in os.environ:
            visible = os.getenv(current_platform.device_control_env_var).split(",")
            n_visible_devices = len(visible)
            n_servers_per_proc = max(1, n_visible_devices // gpus_per_server)
            server_idx_offset = min(list(map(int, visible))) // gpus_per_server
        else:
            n_servers_per_proc = n_servers_per_node
            server_idx_offset = 0

        # Separate ports used by each server in the same node
        # ports range (10000, 32767)
        ports_per_server = 22767 // n_servers_per_node
        launch_server_args = []
        server_addresses = []
        base_random_seed = self.config.random_seed
        for server_local_idx in range(
            server_idx_offset, server_idx_offset + n_servers_per_proc
        ):
            port_range = (
                server_local_idx * ports_per_server + 10000,
                (server_local_idx + 1) * ports_per_server + 10000,
            )
            server_port, dist_init_port = find_free_ports(2, port_range)

            if cross_nodes:
                n_nodes = n_nodes_per_server
                dist_init_addr = f"{master_addr}:{master_port}"
            else:
                n_nodes = 1
                dist_init_addr = f"localhost:{dist_init_port}"

            host_ip = gethostip()

            base_gpu_id = (server_local_idx - server_idx_offset) * gpus_per_server
            config = deepcopy(self.config)
            config.random_seed = base_random_seed + server_local_idx
            cmd = SGLangConfig.build_cmd(
                config,
                tp_size=self.allocation_mode.gen.tp_size,
                base_gpu_id=base_gpu_id,
                host=host_ip,
                port=server_port,
                dist_init_addr=dist_init_addr,
                n_nodes=n_nodes,
                node_rank=node_rank,
                pp_size=self.allocation_mode.gen.pp_size,
            )
            launch_server_args.append((cmd, host_ip, server_port, node_rank))
            server_addresses.append(f"http://{format_hostport(host_ip, server_port)}")

        with ThreadPoolExecutor(max_workers=n_servers_per_proc) as executor:
            server_iterator = executor.map(
                lambda args: self.launch_one_server(*args), launch_server_args
            )
            # Collect all server processes
            self.server_processes = list(server_iterator)

        # Monitor server processes
        self._monitor_server_processes(server_addresses)

    def launch_one_server(self, cmd, host_ip, server_port, node_rank):
        server_process = launch_server_cmd(cmd)
        wait_for_server(f"http://{format_hostport(host_ip, server_port)}")
        if node_rank == 0:
            name = names.gen_servers(self.experiment_name, self.trial_name)
            name_resolve.add_subentry(name, f"{host_ip}:{server_port}")
        logger.info(
            f"SGLang server launched at: http://{format_hostport(host_ip, server_port)}"
        )
        return server_process


def launch_sglang_server(argv):
    config, _ = parse_cli_args(argv)
    config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
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
    assert allocation_mode.gen_backend == "sglang"

    # Get CPU per GPU from rollout scheduling spec
    rollout_spec = get_scheduling_spec(config.rollout)

    if config.rollout.return_routed_experts:
        config.sglang.enable_return_routed_experts = True

    sglang_server = SGLangServerWrapper(
        config.experiment_name,
        config.trial_name,
        config.sglang,
        allocation_mode,
        n_gpus_per_node=config.cluster.n_gpus_per_node,
        cpu_per_gpu=rollout_spec.cpu,
    )
    sglang_server.run()


def main(argv):
    try:
        launch_sglang_server(argv)
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        kill_process_tree(graceful=True)


if __name__ == "__main__":
    main(sys.argv[1:])
