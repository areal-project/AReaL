# SPDX-License-Identifier: Apache-2.0

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.utils.network import find_free_ports, gethostip


def get_placement_group_master_ip_and_port(
    placement_group: PlacementGroup, placement_group_bundle_index: int = 0
):
    def _master_ip_and_port():
        host_ip = gethostip()
        port = find_free_ports(1, (10000, 32767))[0]
        return host_ip, port

    future = ray.remote(
        num_cpus=1,
        num_gpus=0,
        memory=10 * 1024 * 1024,  # Convert MB to bytes
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=placement_group_bundle_index,
        ),
    )(_master_ip_and_port).remote()
    return ray.get(future)


def create_resource_spec(device, cpu: int, gpu: int, mem_in_bytes: int):
    res = {"num_cpus": cpu, "memory": mem_in_bytes}
    if device == "GPU":
        res["num_gpus"] = float(gpu)
    elif device != "CPU":
        res["resources"] = {device: float(gpu)}
    return res
