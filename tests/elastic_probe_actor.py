# SPDX-License-Identifier: Apache-2.0
"""GPU-only instrumented Actor for elastic lifecycle verification, never a recipe."""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
    ShardedTensorFactory,
)

from areal.engine import MegatronPPOActor


class ProbeActor(MegatronPPOActor):
    def initialize(self, *args, **kwargs):
        self.hf_load_seconds = 0.0
        self.hf_load_calls = 0
        super().initialize(*args, **kwargs)

    def _load_model_from_hf(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return super()._load_model_from_hf(*args, **kwargs)
        finally:
            self.hf_load_seconds += time.monotonic() - start
            self.hf_load_calls += 1

    def load(self, meta):
        start = time.monotonic()
        super().load(meta)
        self.dcp_load_seconds = time.monotonic() - start

    def get_device_stats(self):
        """All-rank audit RPC; hash actual export state, excluding marked padding."""
        start = time.monotonic()
        state = self.checkpointer.generate_state_dict(with_rng=False, is_loading=False)
        records = []
        coverage = {
            name: {"tensors": 0, "numel": 0}
            for name in ("model", "master", "exp_avg", "exp_avg_sq")
        }

        def walk(value, path):
            if isinstance(value, dict):
                padding = value.get("padding", False)
                if isinstance(padding, LocalNonpersistentObject):
                    padding = padding.unwrap()
                if padding is True:
                    return
                for key in sorted(value, key=str):
                    walk(value[key], path + "/" + str(key))
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    walk(item, path + "/" + str(index))
            elif isinstance(value, ShardedTensorFactory):
                walk(value.build(), path)
            elif isinstance(value, ShardedTensor):
                metadata = {
                    key: getattr(value, key)
                    for key in (
                        "key",
                        "dtype",
                        "local_shape",
                        "global_shape",
                        "global_offset",
                        "axis_fragmentations",
                        "prepend_axis_num",
                        "replica_id",
                    )
                }
                metadata["dtype"] = str(metadata["dtype"])
                if value.flattened_range is not None:
                    metadata["flattened_range"] = [
                        value.flattened_range.start,
                        value.flattened_range.stop,
                        value.flattened_range.step,
                    ]
                records.append([path, metadata])
                walk(value.data, path)
            elif isinstance(value, ShardedObject):
                walk(value.data, path)
            elif isinstance(value, LocalNonpersistentObject):
                walk(value.unwrap(), path)
            elif isinstance(value, torch.Tensor):
                data = value.detach().cpu().contiguous()
                digest = hashlib.sha256(
                    data.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                records.append([path, str(data.dtype), list(data.shape), digest])
                category = (
                    "model"
                    if path.startswith("/model")
                    else next(
                        (
                            name
                            for name in ("exp_avg_sq", "exp_avg")
                            if path.endswith("/" + name)
                        ),
                        "master" if path.endswith("/param") else None,
                    )
                )
                if category:
                    coverage[category]["tensors"] += 1
                    coverage[category]["numel"] += data.numel()
            elif isinstance(value, io.BytesIO):
                records.append([path, hashlib.sha256(value.getvalue()).hexdigest()])
            elif value is None or isinstance(value, (str, bool, int, float)):
                records.append([path, value])
            else:
                raise TypeError(f"Unverified state wrapper at {path}: {type(value)}")

        walk(state, "")
        for name, count in coverage.items():
            if not count["numel"]:
                raise ValueError(f"Audit did not cover {name}")
        audit_index = getattr(self, "_audit_index", 0)
        self._audit_index = audit_index + 1
        audit_root = Path(os.environ["AREAL_ELASTIC_PROBE_AUDIT_DIR"])
        audit_root.mkdir(parents=True, exist_ok=True)
        owner = os.environ["AREAL_ELASTIC_PROCESS_OWNER"]
        (audit_root / f"{owner}-rank{self.rank}-{audit_index}.json").write_text(
            json.dumps(records, sort_keys=True)
        )
        return {
            "rank": self.rank,
            "state_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True).encode()
            ).hexdigest(),
            "coverage": coverage,
            "hf_load_calls": self.hf_load_calls,
            "hf_load_seconds": self.hf_load_seconds,
            "dcp_load_seconds": self.dcp_load_seconds,
            "audit_seconds": time.monotonic() - start,
        }

    def optimizer_step(self):
        """Controlled nonzero gradient checks restored Adam state; not a PPO update."""
        self.optimizer_zero_grad()
        for model in self.model:
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.main_grad.fill_(0.001)
        return super().optimizer_step()
