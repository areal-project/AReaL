# SPDX-License-Identifier: Apache-2.0

"""Controller surface for forward-only MOPD teacher scoring."""

from __future__ import annotations

from typing import Any

from areal.infra import TrainController


class MOPDTeacherController(TrainController):
    """Dispatch scoring with pipeline-safe padding and active dummy outputs."""

    def compute_logp_padded(self, data: list[dict[str, Any]]):
        original_size = len(data)
        pp_size = self.parallel_strategy.pp_size
        min_microbatches = max(
            2 * pp_size if pp_size > 1 else 1,
            self.config.mb_spec.n_mbs,
        )
        min_items_per_dp = (
            ((min_microbatches + pp_size - 1) // pp_size)
            * pp_size
            * self.config.mb_spec.granularity
        )
        args, kwargs = self._pad_eval_dispatch_args(
            (data,),
            {},
            group_size=1,
            min_items_per_dp=min_items_per_dp,
            items_per_dp_divisor=pp_size * self.config.mb_spec.granularity,
            active_dummies=True,
        )
        results = self._custom_function_call(
            "compute_logp", *args, rpc_meta={"broadcast": True}, **kwargs
        )
        if results is None:
            return None, []
        return results[:original_size], results[original_size:]

    def assert_mopd_runtime_topology(self) -> None:
        self._custom_function_call(
            "assert_mopd_runtime_topology", rpc_meta={"broadcast": False}
        )
