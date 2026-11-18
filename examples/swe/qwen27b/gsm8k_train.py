# SPDX-License-Identifier: Apache-2.0
"""Run native GSM8K RLVR with the dense Qwen worker and recovery contract."""

import os


def main(args: list[str]) -> None:
    from examples.swe.qwen38_flash_next.train_rl import (
        configure_training_rpc,
        validate_required_recovery,
    )
    from examples.swe.rl.qwen38_27b.qwen27_train_rl import decorate_dense_controller

    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.dataset.gsm8k import get_gsm8k_rl_dataset
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.utils.hf_utils import load_hf_tokenizer

    config, _ = load_expr_config(args, GRPOConfig)
    thinking = os.environ.get("QWEN_GSM8K_ENABLE_THINKING", "false").lower()
    if thinking not in ("true", "false"):
        raise ValueError("QWEN_GSM8K_ENABLE_THINKING must be true or false")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_gsm8k_rl_dataset(
        path=config.train_dataset.path,
        split="train",
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_gsm8k_rl_dataset(
            path=config.valid_dataset.path,
            split="test",
            tokenizer=tokenizer,
            max_length=config.valid_dataset.max_length,
        )
    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=thinking == "true",
    )

    class RecipeTrainer(PPOTrainer):
        def _init_scheduler(self):
            return configure_training_rpc(super()._init_scheduler())

    factory = RemoteSGLangEngine.as_controller
    RemoteSGLangEngine.as_controller = staticmethod(
        lambda *args, **kwargs: decorate_dense_controller(factory(*args, **kwargs))
    )
    try:
        with RecipeTrainer(
            config, train_dataset=train_dataset, valid_dataset=valid_dataset
        ) as trainer:
            validate_required_recovery(trainer.recover_info)
            trainer.train(
                workflow="areal.workflow.rlvr.RLVRWorkflow",
                workflow_kwargs=workflow_kwargs,
                eval_workflow="areal.workflow.rlvr.RLVRWorkflow"
                if valid_dataset is not None
                else None,
                eval_workflow_kwargs={**workflow_kwargs, "gconfig": config.eval_gconfig}
                if valid_dataset is not None
                else None,
            )
    finally:
        RemoteSGLangEngine.as_controller = staticmethod(factory)
