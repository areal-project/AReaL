# SPDX-License-Identifier: Apache-2.0
"""DSpark's native V2 worker with AReaL's retained-action target distribution."""

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    CapCorrectLen,
    FinalizeAcceptLens,
    SoftmaxTemp,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import _get_or_create_chain_verify_buffers
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite


def accept_joint_chain(candidates, target_probs, draft_probs, verify_lens=None):
    bs, width, _ = target_probs.shape
    device = candidates.device
    index, child, sibling, tokens, accept_index, accepted = (
        _get_or_create_chain_verify_buffers(bs=bs, draft_token_num=width, device=device)
    )
    chain_speculative_sampling_triton(
        predicts=tokens,
        accept_index=accept_index,
        accept_token_num=accepted,
        candidates=candidates,
        retrive_index=index,
        retrive_next_token=child,
        retrive_next_sibling=sibling,
        uniform_samples=torch.rand((bs, width - 1), dtype=torch.float32, device=device),
        uniform_samples_for_final_sampling=torch.rand(
            (bs,), dtype=torch.float32, device=device
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    if verify_lens is None:
        trimmed = torch.zeros_like(accepted)
    else:
        accepted, trimmed = CapCorrectLen.execute(
            correct_len=accepted, verify_lens=verify_lens
        )
    pos = accept_index[torch.arange(bs, device=device), accepted.long()].long()
    return accepted, tokens[pos].long(), trimmed


class JointVerifyExecutor(TargetVerifyExecutor):
    """Upstream runs these two phases consecutively on the same scheduler thread."""

    pending_output = None

    def run_non_compact(self, *, batch, sampling_info, **kwargs):
        # The mask sampler rejects logit transforms and builds the scaled
        # target distribution once, during acceptance.
        result = super().run_non_compact(batch=batch, sampling_info=None, **kwargs)
        self.pending_output = (batch, result.logits_output)
        return result

    def run_compact(self, *, batch, sampling_info, **kwargs):
        result, hidden = super().run_compact(batch=batch, sampling_info=None, **kwargs)
        self.pending_output = (batch, result.logits_output)
        return result, hidden

    def accept_and_finalize(
        self,
        *,
        folded_accept,
        bs,
        verify_ids_2d,
        target_logits,
        draft_block,
        sampling_info,
        draft_input,
        layout,
        prefix_lens,
        draft_tokens,
    ):
        if folded_accept or self._simulate_acc_len > 0:
            raise ValueError("AReaL joint verify requires stochastic acceptance")
        batch, output = self.pending_output
        self.pending_output = None
        policy = self.target_worker.model_runner.sampler.prepare_verify_policy(
            target_logits, batch, self.verify_num_draft_tokens
        )
        logits = draft_block.corrected_logits
        draft_probs = SoftmaxTemp.execute(
            logits=logits.flatten(0, 1),
            temperatures=draft_block.temperatures,
            rows_per_request=logits.shape[1],
        ).view_as(logits)
        accepted, bonus, trimmed = accept_joint_chain(
            verify_ids_2d,
            policy.target_probs,
            draft_probs,
            layout.verify_lens if layout is not None else None,
        )
        for value in (accepted, bonus, trimmed):
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE, value)
        lens = FinalizeAcceptLens.execute(
            correct_len=accepted, cap_trim_lens=trimmed, prefix_lens=prefix_lens
        )
        tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=accepted,
            bonus=bonus,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        policy.finish(output, tokens, lens.commit_lens)
        return AcceptOuts(
            correct_len=accepted,
            bonus=bonus,
            cap_trim_lens=lens.cap_trim_lens,
            commit_lens=lens.commit_lens,
            new_seq_lens=lens.new_seq_lens,
            out_tokens=tokens,
        )


class JointDSparkWorker(DSparkWorkerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._verify_executor = JointVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            tp_sync=self._tp_sync,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
        )

    def _forward_decode(self, batch, on_publish, grammar_barrier=None):
        # The executor sets next_token_logprobs from the same support used for
        # rejection. Suppress the parent's independent full-vocab logprob pass.
        return_logprob = batch.return_logprob
        batch.return_logprob = False
        try:
            return super()._forward_decode(batch, on_publish, grammar_barrier)
        finally:
            batch.return_logprob = return_logprob
            self._verify_executor.pending_output = None

    def _forward_prefill(self, batch, on_publish):
        if not batch.forward_mode.is_idle():
            return super()._forward_prefill(batch, on_publish)
        if get_parallel().enable_dp_attention:
            # Join peers' forward collectives. No token or auxiliary hidden
            # states belong to this rank. Qwen's outer VLM wrapper normally
            # unpacks the inner result as a pair, but idle returns one tensor.
            model = self.target_worker.model_runner.model
            capture = getattr(model, "capture_aux_hidden_states", None)
            if capture:
                model.capture_aux_hidden_states = False
            try:
                self.target_worker.forward_batch_generation(
                    batch, is_verify=True, capture_hidden_mode=CaptureHiddenMode.NULL
                )
            finally:
                if capture:
                    model.capture_aux_hidden_states = capture
        return self._decode_idle_result(on_publish=on_publish)
