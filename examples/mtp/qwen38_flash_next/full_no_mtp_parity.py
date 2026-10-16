# SPDX-License-Identifier: Apache-2.0
"""Compare full source-checkpoint backbone forward/loss against mcore-bridge."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from areal.models.mcore.mcore_bridge_adapter import MCoreBridgeAdapter


def _language_model(model):
    while hasattr(model, "module"):
        model = model.module
    return getattr(model, "language_model", model)


def _forward_observation_counts(
    expected, actual, *, expected_microbatches, is_last_stage
):
    """Require every scheduled last-PP microbatch to expose real logits/loss."""
    if expected_microbatches <= 0:
        raise AssertionError("Prepared parity microbatch list is empty")
    required = expected_microbatches if is_last_stage else 0
    if len(expected) != required or len(actual) != required:
        raise AssertionError(
            f"Parity observation count mismatch: last_pp={is_last_stage}, "
            f"required={required}, candidate={len(expected)}, reference={len(actual)}"
        )
    for outputs in (expected, actual):
        if any(logits.numel() == 0 or loss.numel() != 1 for logits, loss in outputs):
            raise AssertionError(
                "Parity observation must contain nonempty logits and scalar loss"
            )
    return [
        required,
        len(expected),
        len(actual),
        sum(logits.numel() for logits, _ in expected),
        sum(logits.numel() for logits, _ in actual),
    ]


def _global_observation_summary(counts):
    required, candidate, reference, candidate_elements, reference_elements = counts
    if required <= 0 or candidate != required or reference != required:
        raise AssertionError(f"Global parity has missing/zero observations: {counts}")
    if candidate_elements <= 0 or candidate_elements != reference_elements:
        raise AssertionError(
            f"Global parity has missing/mismatched logit elements: {counts}"
        )
    return {
        "expected_microbatches_rank_sum": required,
        "candidate_observed_microbatches_rank_sum": candidate,
        "reference_observed_microbatches_rank_sum": reference,
        "compared_microbatches_rank_sum": candidate,
        "compared_logit_elements_rank_sum": candidate_elements,
    }


@torch.no_grad()
def compare_full_no_mtp(actor, batch) -> None:
    """Use the actual loaded checkpoint, TP/PP/EP topology and a short fixed input."""
    from megatron.core import parallel_state

    candidate = actor.model
    was_training = candidate[0].training
    controls = None
    policy = yaml.safe_load(
        (Path(__file__).parent / "acceptance_tolerances.yaml").read_text()
    )["comparisons"]["no_mtp_forward"]
    try:
        actor.eval()
        # Truncate real samples solely for short-sequence forward comparison.
        fixed = {
            key: value[:, :512].clone()
            if isinstance(value, torch.Tensor) and value.ndim == 2
            else value
            for key, value in batch.items()
        }
        outputs = []

        def capture(output, inputs):
            logprobs = actor._compute_forward_result(output, inputs)
            mask = torch.roll(inputs["loss_mask"].bool(), shifts=-1, dims=-1)
            loss = -(logprobs * mask).sum() / mask.sum().clamp_min(1)
            outputs.append(
                (
                    output.detach().float().cpu().clone(),
                    loss.detach().float().cpu().clone(),
                )
            )
            return None

        prepared_microbatches = actor._prepare_mb_list(
            fixed, allow_transport_padding=False
        )
        expected_microbatches = len(prepared_microbatches)
        actor.forward_backward_batch(prepared_microbatches, capture, forward_only=True)
        reference_outputs = outputs
        outputs = []
        # Only the frozen source model is moved. This process exits after parity:
        # do not reuse its DDP/optimizer buffers for subsequent training or export.
        for model in candidate:
            _language_model(model).cpu()
        gc.collect()
        torch.cuda.empty_cache()
        reference = MCoreBridgeAdapter(
            os.environ["MODEL_PATH"],
            dtype=torch.bfloat16,
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            expert_model_parallel_size=8,
            expert_tensor_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            gradient_checkpointing=False,
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
            distribute_saved_activations=None,
            recompute_modules=None,
            language_model_only=True,
            freeze_ple_table=True,
            transformer_config_overrides={
                "mtp_num_layers": None,
                "mtp_unroll_steps": 0,
            },
        )
        controls = reference.get_model(wrap_with_ddp=False, ddp_config={})
        if len(controls) != len(candidate):
            raise AssertionError("Reference/candidate virtual chunk counts differ")
        for actual, control in zip(candidate, controls, strict=True):
            # Source weights have already been converted by Megatron-Bridge.
            # Independent complete HF roundtrip checks their provenance separately.
            state = {
                name: value
                for name, value in _language_model(actual).state_dict().items()
                if not name.startswith("mtp.")
            }
            _language_model(control).load_state_dict(state, strict=True)
            control.requires_grad_(False)
            control.eval()
        actor.model = controls
        prepared_control = actor._prepare_mb_list(fixed, allow_transport_padding=False)
        if len(prepared_control) != expected_microbatches:
            raise AssertionError(
                "Reference/candidate prepared microbatch counts differ"
            )
        actor.forward_backward_batch(prepared_control, capture, forward_only=True)
        counts = torch.tensor(
            _forward_observation_counts(
                reference_outputs,
                outputs,
                expected_microbatches=expected_microbatches,
                is_last_stage=parallel_state.is_pipeline_last_stage(
                    ignore_virtual=True
                ),
            ),
            device=actor.device,
            dtype=torch.int64,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        observation_summary = _global_observation_summary(counts.cpu().tolist())
        max_abs = max_rel = loss_abs = loss_rel = 0.0
        for (expected, expected_loss), (actual, actual_loss) in zip(
            reference_outputs, outputs, strict=True
        ):
            torch.testing.assert_close(
                actual, expected, atol=policy["logits_atol"], rtol=policy["logits_rtol"]
            )
            torch.testing.assert_close(
                actual_loss,
                expected_loss,
                atol=policy["loss_atol"],
                rtol=policy["loss_rtol"],
            )
            delta = (actual - expected).abs()
            max_abs = max(max_abs, float(delta.max()))
            max_rel = max(
                max_rel,
                float(
                    (
                        delta
                        / expected.abs().clamp_min(torch.finfo(torch.float32).tiny)
                    ).max()
                ),
            )
            loss_abs = max(loss_abs, float((actual_loss - expected_loss).abs()))
            loss_rel = max(
                loss_rel,
                float(
                    (actual_loss - expected_loss).abs()
                    / expected_loss.abs().clamp_min(torch.finfo(torch.float32).tiny)
                ),
            )
        metrics = torch.tensor(
            [max_abs, max_rel, loss_abs, loss_rel],
            device=actor.device,
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "full_no_mtp_parity": "PASS",
                        **observation_summary,
                        "logits_max_abs": metrics[0].item(),
                        "logits_max_rel": metrics[1].item(),
                        "loss_max_abs": metrics[2].item(),
                        "loss_max_rel": metrics[3].item(),
                        "atol": policy["logits_atol"],
                        "rtol": policy["logits_rtol"],
                        "sequence_length": 512,
                        "topology": "TP8 PP2 EP8 CP1",
                    }
                ),
                flush=True,
            )
    finally:
        actor.model = candidate
        actor.train(was_training)
        del controls
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    import sys

    from areal import SFTTrainer
    from areal.api.cli_args import SFTConfig, load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer

    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    if config.recover.mode != "disabled":
        raise ValueError("Full parity requires pristine HF initialization")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    with SFTTrainer(config, train_dataset=dataset, valid_dataset=None) as trainer:
        batch = trainer.train_dataloader.collate_fn(
            [trainer.train_dataloader.dataset[i] for i in range(4)]
        )
        compare_full_no_mtp(trainer.actor, batch)


if __name__ == "__main__":
    main()
