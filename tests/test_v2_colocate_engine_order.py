# SPDX-License-Identifier: Apache-2.0
"""Engine index parity decides which training DP replica feeds each engine."""

import pytest


def _assign(devices, infer_by_device, node_minor):
    """Mirror the gateway's device/server ordering and base-rank handout."""

    def key(device):
        ip, device_id = device
        _url, local, _iw = infer_by_device[device]
        return (device_id - local, ip, local)

    paired = sorted(devices, key=key) if node_minor else sorted(devices)
    servers = {}
    for ip, device_id in devices:
        url, local, iw = infer_by_device[(ip, device_id)]
        servers[url] = (ip, device_id - local, iw)
    order = sorted(
        servers.items(),
        key=lambda kv: (kv[1][1], kv[1][0]) if node_minor else (kv[1][0], kv[1][1]),
    )
    base = {}
    nxt = 0
    for url, (_ip, _base_gpu, iw) in order:
        base[url] = nxt
        nxt += iw
    return paired, base


@pytest.fixture
def two_node_four_engines():
    """2 nodes x 8 GPUs, 4 SGLang servers of TP=4."""
    ips = ("10.0.0.1", "10.0.0.2")
    devices, infer_by_device = [], {}
    for ip in ips:
        for base_gpu in (0, 4):
            url = f"http://{ip}:{8000 + base_gpu}"
            for local in range(4):
                dev = (ip, base_gpu + local)
                devices.append(dev)
                infer_by_device[dev] = (url, local, 4)
    return devices, infer_by_device, ips


def test_node_minor_keeps_contiguous_coverage(two_node_four_engines):
    devices, infer_by_device, _ = two_node_four_engines

    paired, base = _assign(devices, infer_by_device, node_minor=True)

    for rank, dev in enumerate(paired):
        url, local, _iw = infer_by_device[dev]
        assert base[url] + local == rank


def test_node_minor_gives_same_node_engines_one_replica(two_node_four_engines):
    devices, infer_by_device, ips = two_node_four_engines

    _, base = _assign(devices, infer_by_device, node_minor=True)

    per_node = {ip: set() for ip in ips}
    for dev, (url, _local, iw) in infer_by_device.items():
        per_node[dev[0]].add((base[url] // iw) % 2)
    assert per_node[ips[0]] == {0}
    assert per_node[ips[1]] == {1}


def test_default_order_splits_each_node_across_replicas(two_node_four_engines):
    devices, infer_by_device, ips = two_node_four_engines

    _, base = _assign(devices, infer_by_device, node_minor=False)

    per_node = {ip: set() for ip in ips}
    for dev, (url, _local, iw) in infer_by_device.items():
        per_node[dev[0]].add((base[url] // iw) % 2)
    assert per_node[ips[0]] == {0, 1}
    assert per_node[ips[1]] == {0, 1}
