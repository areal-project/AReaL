import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from examples.swe.utils import SWEPPOConfig

from areal.api.cli_args import to_structured_cfg
from areal.utils import checkpoint_pointer

root = Path(os.environ["EXPERIMENT_ROOT"])
cfg = OmegaConf.to_object(
    to_structured_cfg(
        OmegaConf.load(os.environ.get("QWEN_CONFIG", str(root / "train.yaml"))),
        SWEPPOConfig,
    )
)
assert cfg.trial_name == os.environ["QWEN_ARENA_TRIAL"]
assert (
    cfg.stats_logger.trial_name
    == cfg.recover.trial_name
    == cfg.saver.trial_name
    == cfg.trial_name
)
assert cfg.actor.scheduling_spec[0].mount == os.environ["QWEN_MOUNTS"]
assert cfg.rollout.scheduling_spec[0].mount == os.environ["QWEN_MOUNTS"]
assert cfg.actor.temperature == cfg.gconfig.temperature
assert cfg.train_dataset.batch_size == cfg.rollout.consumer_batch_size
checkpoint = (
    Path(cfg.recover.fileroot)
    / "checkpoints"
    / os.environ["USER"]
    / cfg.experiment_name
    / cfg.trial_name
)

source = checkpoint_pointer.resolve_checkpoint(str(checkpoint), None)
require_recovery = os.environ.get("QWEN_REQUIRE_RECOVER", "0") == "1"
if require_recovery and source is None:
    raise RuntimeError("Recovery is required, but this trial has no valid checkpoint")
if checkpoint.exists() and any(checkpoint.iterdir()) and source is None:
    raise RuntimeError(
        "Checkpoint directory is nonempty but no valid resume pointer exists"
    )
for key in (
    "QWEN_ISOLATE_REQUEST_CACHE",
    "QWEN_QSA_STABLE_TOPK",
    "QWEN_BATCH_REPLAY_PATH",
    "QWEN_BATCH_REPLAY_PATHS",
    "QWEN_BATCH_SNAPSHOT_DIR",
):
    assert not os.environ.get(key), f"Unexpected diagnostic environment {key}"
assert cfg.actor.path == cfg.tokenizer_path == os.environ["QWEN_MODEL"]
assert cfg.sglang.context_length == cfg.gconfig.max_tokens
result = {
    "schema_valid": True,
    "structured_config_validated": True,
    "trial": cfg.trial_name,
    "sampling": {
        "top_p": cfg.gconfig.top_p,
        "top_k": cfg.gconfig.top_k,
        "temperature": cfg.gconfig.temperature,
    },
    "reward_normalization_use_std": cfg.gconfig.reward_normalization_use_std,
    "cache_enabled": not cfg.sglang.disable_radix_cache,
    "r3": cfg.rollout.return_routed_experts,
    "lm_head": {
        "enable_chunked_logits": cfg.actor.megatron.enable_chunked_logits,
        "lm_head_loss_chunk_size": cfg.actor.megatron.lm_head_loss_chunk_size,
    },
    "enable_memory_saver": cfg.sglang.enable_memory_saver,
    "disable_cuda_graph": cfg.sglang.disable_cuda_graph,
    "max_concurrent_rollouts": cfg.rollout.max_concurrent_rollouts,
    "n_samples": cfg.gconfig.n_samples,
    "batch_size": cfg.train_dataset.batch_size,
    "checkpoint_root": str(checkpoint),
    "resume_checkpoint_available": source is not None,
    "resume_required": require_recovery,
    "recover_mode": cfg.recover.mode,
    "recover_freq_steps": cfg.recover.freq_steps,
    "recover_freq_secs": cfg.recover.freq_secs,
    "images": {
        "actor": os.environ["QWEN_ACTOR_IMAGE"],
        "rollout": os.environ["QWEN_ROLLOUT_IMAGE"],
    },
    "no_trainer_instantiated": True,
}
(root / "structured-config-validation.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print("STRUCTURED_CONFIG_REVIEW_PASS " + json.dumps(result), flush=True)
