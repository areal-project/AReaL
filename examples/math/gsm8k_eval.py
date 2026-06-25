import sys

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config, vLLMConfig
from areal.dataset import get_custom_dataset
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler
from areal.utils import logging, seeding
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats

logger = logging.getLogger("GSM8KEval")


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    logging.setup_file_logging(f"{config.cluster.fileroot}/eval.log")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key="eval")

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")

    # Initialize scheduler
    cfg = config.scheduler
    if cfg.type == "local":
        scheduler = LocalScheduler(exp_config=config)
    elif cfg.type == "ray":
        scheduler = RayScheduler(exp_config=config)
    elif cfg.type == "slurm":
        scheduler = SlurmScheduler(exp_config=config)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.type}")

    # Load evaluation dataset
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )

    # Initialize RolloutController
    config.rollout.max_head_offpolicyness = int(1e12)

    if rollout_alloc.backend == "sglang":
        engine_cls = RemoteSGLangEngine
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=rollout_alloc.parallel.tp_size,
            base_gpu_id=0,
        )
    elif rollout_alloc.backend == "vllm":
        engine_cls = RemotevLLMEngine
        server_args = vLLMConfig.build_args(
            vllm_config=config.vllm,
            tp_size=rollout_alloc.parallel.tp_size,
            pp_size=rollout_alloc.parallel.pp_size,
        )
    else:
        raise ValueError(f"Invalid backend: {rollout_alloc.backend}")

    eval_rollout = engine_cls.as_controller(config.rollout, scheduler)

    try:
        eval_rollout.initialize(
            role="eval-rollout",
            server_args=server_args,
        )

        # Create evaluation workflow
        workflow = "areal.workflow.rlvr.RLVRWorkflow"
        workflow_kwargs = dict(
            reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=False,
        )

        # Submit all evaluation tasks
        cnt = 0
        for data in valid_dataloader:
            for item in data:
                eval_rollout.submit(
                    item,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    group_size=config.gconfig.n_samples,
                )
                cnt += 1

        eval_rollout.wait(cnt, timeout=None)
        eval_stats = eval_rollout.export_stats()

        # Print and log results
        logger.info(f"Evaluation Results: {tabulate_stats(eval_stats)}")
    finally:
        eval_rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
