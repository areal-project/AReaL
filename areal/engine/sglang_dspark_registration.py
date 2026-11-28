# SPDX-License-Identifier: Apache-2.0
"""Register AReaL's stochastic target verify worker on stock SGLang 0.5.20."""

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

from areal.engine.sglang_sampling_mask_plugin import JOINT_DSPARK

_DSPARK = SpeculativeAlgorithm.DSPARK


class JointDSparkAlgorithm(CustomSpecAlgo):
    is_dspark = _DSPARK.is_dspark
    is_dflash_family = _DSPARK.is_dflash_family
    supports_target_verify_for_draft = _DSPARK.supports_target_verify_for_draft
    supports_ragged_verify = _DSPARK.supports_ragged_verify
    supports_grammar_overlap = _DSPARK.supports_grammar_overlap
    supports_mixed_chunk = _DSPARK.supports_mixed_chunk
    need_topk = _DSPARK.need_topk
    carries_draft_hidden_states = _DSPARK.carries_draft_hidden_states
    create_future_map = _DSPARK.create_future_map
    build_disagg_draft_input = _DSPARK.build_disagg_draft_input
    get_num_tokens_per_req_for_target_verify = (
        _DSPARK.get_num_tokens_per_req_for_target_verify
    )

    def handle_server_args(self, server_args):
        # Built-ins declare defaults; external algorithms assign them through
        # SGLang's recording proxy. Preserve the upstream validation and order.
        class NativeDefaults:
            def __init__(self):
                self._resolved_overrides = []

            def __getattr__(self, name):
                return getattr(server_args, name)

        defaults = NativeDefaults()
        _DSPARK.handle_server_args(defaults)
        for _, fields in defaults._resolved_overrides:
            for name, value in fields.items():
                setattr(server_args, name, value)


def register_dspark():
    @SpeculativeAlgorithm.register(
        JOINT_DSPARK, supports_overlap=True, spec_class=JointDSparkAlgorithm
    )
    def worker_class(server_args):
        from areal.engine.sglang_dspark_worker import JointDSparkWorker

        return JointDSparkWorker
