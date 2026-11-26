# SPDX-License-Identifier: Apache-2.0
"""Live native vLLM transport, dense probability oracle and FSDP optimizer smoke.

Requires an SC-configured FP32 server (TRITON_F32_DEFAULT=ieee,
NVIDIA_TF32_OVERRIDE=0) and the same local checkpoint.
Use a separate GPU for this process; the server remains fixed across two updates.
"""

import argparse
import asyncio
import copy
import gc
import json
import os
from types import SimpleNamespace
from urllib.parse import urlsplit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from areal import workflow_context
from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    RejectionSamplingConfig,
)
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.vllm_remote import VLLMBackend
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn
from areal.utils import logging, seeding
from areal.utils.data import concat_batch
from areal.utils.score_centering import (
    score_centering_logprobs,
    score_centering_tis_weights,
)
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("VLLMScoreCenteringTest")


def smoke_reward(*args, smoke_reward: float, **kwargs) -> float:
    return smoke_reward


async def collect(url, tokenizer):
    engine = RemoteInfEngine(
        InferenceEngineConfig(request_retries=1), backend=VLLMBackend()
    )
    engine.choose_server = lambda: urlsplit(url).netloc
    engine.workflow_executor = SimpleNamespace(is_paused=lambda: False)
    engine.get_version = lambda: 0
    responses, trajectories = [], []
    generate = engine.agenerate

    async def record(req):
        response = await generate(req)
        responses.append((req.gconfig.score_centering_top_k, response))
        return response

    engine.agenerate = record
    try:
        for width in (1, 8, 128):
            for seed in (17, 29):
                workflow = RLVRWorkflow(
                    reward_fn=smoke_reward,
                    tokenizer=tokenizer,
                    gconfig=GenerationHyperparameters(
                        max_new_tokens=32,
                        max_tokens=256,
                        top_k=-1,
                        score_centering_top_k=width,
                        seed=seed,
                    ),
                )
                trajectory = await workflow.arun_episode(
                    engine,
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Use 2, 3, 7, 8 exactly once with arithmetic to make 24. Explain your reasoning.",
                            }
                        ],
                        "smoke_reward": 1.0 if seed == 17 else -1.0,
                    },
                )
                if width == 128:
                    trajectories.append(trajectory)
    finally:
        await (await workflow_context.get_aiohttp_session()).close()
    return responses, trajectories


def check_dense_objective(logits, response, cap):
    """Independent full-vocabulary surrogate using actual sampled q, including tails."""
    count = response.output_len
    x = logits[:count].double().detach()
    x = (
        x + 2.0 * torch.sin(torch.arange(x.shape[-1], device=x.device))
    ).requires_grad_()
    other = x.detach().clone().requires_grad_()
    labels = torch.tensor(response.output_tokens[:count], device=x.device)
    ids = torch.tensor(response.sampler_head_ids[:count], device=x.device)
    qh = torch.tensor(
        response.sampler_head_logprobs[:count], device=x.device, dtype=torch.float64
    )
    qa = torch.tensor(
        response.output_logprobs[:count], device=x.device, dtype=torch.float64
    )
    advantages = torch.linspace(-1, 1, count, device=x.device, dtype=torch.float64)
    lp, _, correction = score_centering_logprobs(x, labels, ids, qh, tis_cap=cap)
    weight = 1 if cap is None else score_centering_tis_weights(lp, qa, cap)
    actual = (advantages * (correction - weight * lp)).mean()
    dense_lp = other.log_softmax(-1)
    with torch.no_grad():
        p = dense_lp.exp()
        rho = (1 - qh.exp().sum(-1)).clamp_min(0) / (
            1 - p.gather(-1, ids).sum(-1)
        ).clamp_min(1e-6)
        q = (rho[:, None] * p).scatter(-1, ids, qh.exp())
        alpha = rho if cap is None else (cap * rho).clamp_max(1)
        u = q if cap is None else torch.minimum(p, cap * q)
        coefficients = u - alpha[:, None] * p
        sample_weight = (
            1
            if cap is None
            else (p.gather(-1, labels[:, None]).squeeze(-1) / qa.exp()).clamp_max(cap)
        )
    expected = (
        advantages
        * (
            (coefficients * dense_lp).sum(-1)
            - sample_weight * dense_lp.gather(-1, labels[:, None]).squeeze(-1)
        )
    ).mean()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-9, atol=1e-10)
    return {
        "tail_samples": (~(ids == labels[:, None]).any(-1)).sum().item(),
        "clipped": (sample_weight == cap).sum().item() if cap is not None else 0,
        "unclipped": (sample_weight < cap).sum().item() if cap is not None else 0,
    }


def check_probabilities(model_path, responses):
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    results = []
    for width, response in responses:
        assert response.output_len > 0
        seq = torch.tensor(
            [response.input_tokens + response.output_tokens[:-1]], device="cuda"
        )
        with torch.no_grad():
            logits = model(seq).logits[0, response.input_len - 1 :].float()
            lp = logits.log_softmax(-1)
            ids = torch.tensor(response.sampler_head_ids, device="cuda")
            labels = torch.tensor(response.output_tokens, device="cuda")
            actual = torch.tensor(response.sampler_head_logprobs, device="cuda")
            samples = torch.tensor(response.output_logprobs, device="cuda")
            head_error = (lp.gather(-1, ids) - actual).abs()
            sample_error = (lp.gather(-1, labels[:, None]).squeeze(-1) - samples).abs()
            # Use FP32 serving for this numerical oracle. BF16 serving/training
            # kernel differences are legitimate off-policy drift, not parser error.
            torch.testing.assert_close(lp.gather(-1, ids), actual, rtol=0, atol=1e-4)
            torch.testing.assert_close(
                lp.gather(-1, labels[:, None]).squeeze(-1), samples, rtol=0, atol=1e-4
            )
            torch.testing.assert_close(
                lp.gather(-1, ids).sort(-1, descending=True).values,
                lp.topk(width, dim=-1).values,
                rtol=0,
                atol=2e-4,
            )
            outside = (~(ids == labels[:, None]).any(-1)).sum().item()
        for cap in (None, 2.0):
            coverage = check_dense_objective(logits, response, cap)
        results.append(
            dict(
                width=width,
                tokens=response.output_len,
                outside_head=outside,
                head_max_error=head_error.max().item(),
                head_mean_error=head_error.mean().item(),
                sampled_max_error=sample_error.max().item(),
                oracle_coverage=coverage,
            )
        )
    assert sum(item["outside_head"] for item in results) > 0
    assert sum(item["oracle_coverage"]["clipped"] for item in results) > 0
    assert sum(item["oracle_coverage"]["unclipped"] for item in results) > 0
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def train(model_path, raw, tis_cap):
    config = PPOActorConfig(
        backend="fsdp:d1",
        path=model_path,
        experiment_name="vllm-score-centering-test",
        trial_name="smoke",
        use_score_centering=True,
        recompute_logprob=False,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=tis_cap)
        if tis_cap is not None
        else None,
        dtype="bfloat16",
        disable_dropout=True,
        kl_ctl=0,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=256, n_mbs=2),
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0),
    )
    engine = FSDPEngine(config)
    engine.create_process_group(ModelAllocation.from_str(config.backend).parallel)
    engine.initialize(
        None, FinetuneSpec(total_train_epochs=1, dataset_size=2, train_batch_size=2)
    )
    actor = PPOActor(config, engine)
    stats = []
    try:
        seen = []

        def loss_fn(logprobs, entropy, inputs, *, sc_correction=None, **kwargs):
            assert sc_correction is not None and torch.isfinite(sc_correction).all()
            seen.append(sc_correction.detach().abs().max().item())
            return grpo_loss_fn(
                logprobs,
                entropy,
                inputs,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
                use_score_centering=True,
                sc_correction=sc_correction,
                rejection_sampling=config.rejection_sampling,
                **kwargs,
            )

        engine.train()
        for _ in range(2):
            batch, _ = concat_batch(actor.compute_advantages(copy.deepcopy(raw)))
            batch = {
                key: value.to(engine.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            parameter = next(engine.model.parameters())
            before = parameter.to_local().detach().clone()
            result = engine.train_batch(
                batch, loss_fn, lambda data: data["loss_mask"].count_nonzero()
            )
            delta = (
                (parameter.to_local().detach().float() - before.float())
                .abs()
                .max()
                .item()
            )
            assert result["update_successful"] == 1 and result["grad_norm"] > 0
            assert torch.isfinite(torch.tensor(result["grad_norm"])) and delta > 0
            stats.append(dict(grad_norm=result["grad_norm"], max_parameter_delta=delta))
        assert seen and max(seen) > 0
    finally:
        engine.destroy()
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--tis-cap", type=float, default=None)
    args = parser.parse_args()
    assert int(os.environ["WORLD_SIZE"]) == 1
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seeding.set_random_seed(42, "vllm-score-centering-test")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    responses, raw = asyncio.run(collect(args.server_url, tokenizer))
    probability_checks = check_probabilities(args.model, responses)
    logger.info("PROBABILITY_CHECKS " + json.dumps(probability_checks))
    updates = train(args.model, raw, args.tis_cap)
    logger.info(
        "PASS "
        + json.dumps(
            dict(
                tis_cap=args.tis_cap,
                probability_checks=probability_checks,
                updates=updates,
            )
        )
    )


if __name__ == "__main__":
    main()
