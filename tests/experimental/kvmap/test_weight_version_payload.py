# SPDX-License-Identifier: Apache-2.0
"""SGLang weight-update payloads carry the policy version only when the meta has one."""

from areal.api.io_struct import ParamSpec, WeightUpdateMeta
from areal.engine.sglang_remote import SGLangBackend


def _meta(version):
    meta = WeightUpdateMeta(type="xccl", nccl_group_name="g", nccl_master_address="127.0.0.1", nccl_master_port=1)
    return meta.with_version(version) if version is not None else meta


def test_distributed_update_payload_includes_version_when_present():
    backend = SGLangBackend()
    specs = [ParamSpec(name="w", shape=(1,), dtype="bfloat16")]
    with_version = backend.build_distributed_weight_update_requests(_meta(7), specs).requests[0].payload
    without = backend.build_distributed_weight_update_requests(_meta(None), specs).requests[0].payload
    assert with_version["weight_version"] == "7" and with_version["abort_all_requests"] is True
    assert "weight_version" not in without
