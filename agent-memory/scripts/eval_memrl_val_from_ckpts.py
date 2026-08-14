#!/usr/bin/env python3
"""Batch-evaluate MemRL checkpoints on the validation set.

Loads each section-level snapshot (1..10) from the MemRL experiment dir,
restores the memory state, and runs inference-only evaluation on the val set
(150 samples). No training, no memory update — just retrieve + inject + solve.

Outputs per-epoch val SR to stdout + a summary JSON.
"""
import json
import logging
import os
import sys
from pathlib import Path

# --- Config ---
PROJECT = Path("/storage/openpsi/users/yl/agent-memory/MemRL")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "3rdparty" / "LifelongAgentBench"))

EXP_DIR = Path("/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect/exp_llb_os_memrl_haiku_20260711-022606")
SNAPSHOT_DIR = EXP_DIR / "snapshot"
CONFIG_PATH = PROJECT / "configs" / "rl_llb_os_memrl_haiku.yaml"
VAL_FILE = PROJECT / "data" / "llb" / "os_interaction_val.json"
RESULTS_PATH = EXP_DIR / "val_eval_results.json"

os.environ.setdefault("MEMRL_OS_SANDBOX", "1")
os.environ.setdefault("MEMRL_OS_BACKEND", "local")
os.environ.setdefault("MEMRL_EMBED_THROTTLE", "1.0")
os.environ.setdefault("MEMRL_LLM_MIN_INTERVAL", "0.8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("val_eval")


def get_section_snapshots():
    """Find section-level snapshots (integer names only, skip batch ckpts like 3_b9)."""
    snaps = []
    for d in sorted(SNAPSHOT_DIR.iterdir()):
        if d.is_dir() and d.name.isdigit():
            snaps.append((int(d.name), d))
    return sorted(snaps)


def main():
    from memrl.configs.config import MempConfig
    from memrl.service.memory_service import MemoryService
    from memrl.service.strategies import StrategyConfiguration, BuildStrategy, RetrieveStrategy, UpdateStrategy

    config = MempConfig.from_yaml(str(CONFIG_PATH))

    # Load val dataset
    val_data = json.loads(VAL_FILE.read_text())
    logger.info("Val dataset: %d samples", len(val_data))

    snapshots = get_section_snapshots()
    if not snapshots:
        logger.error("No section snapshots found in %s", SNAPSHOT_DIR)
        sys.exit(1)
    logger.info("Found %d section snapshots: %s", len(snapshots), [s[0] for s in snapshots])

    results = []

    for section_num, snap_dir in snapshots:
        logger.info("=" * 60)
        logger.info("Evaluating snapshot %d (section %d) on val set...", section_num, section_num)

        # Build a fresh MemoryService and load the snapshot
        mos_config_path = str(PROJECT / config.memory.mos_config_path)
        from memrl.providers.llm import OpenAILLM
        from memrl.providers.embedding import OpenAIEmbedding

        llm_provider = OpenAILLM(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            default_temperature=config.llm.temperature,
            default_max_tokens=config.llm.max_tokens,
        )
        embedding_provider = OpenAIEmbedding(
            api_key=config.embedding.api_key,
            base_url=config.embedding.base_url,
            model=config.embedding.model,
            dimension=config.embedding.dimension,
        )
        from memrl.service.strategies import get_rl_config
        rl_config = get_rl_config(config)

        memory_service = MemoryService(
            mos_config_path=mos_config_path,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            strategy_config=StrategyConfiguration(
                BuildStrategy(config.memory.build_strategy),
                RetrieveStrategy(config.memory.retrieve_strategy),
                UpdateStrategy(config.memory.update_strategy),
            ),
            user_id=config.memory.user_id + "_val_eval",  # separate user to not pollute
            num_workers=config.experiment.batch_size,
            max_keywords=config.memory.max_keywords,
            add_similarity_threshold=config.memory.add_similarity_threshold,
            enable_value_driven=config.experiment.enable_value_driven,
            rl_config=rl_config,
            vector_dimension=config.embedding.dimension,
            sim_norm_mean=getattr(config.memory, "sim_norm_mean", None),
            sim_norm_std=getattr(config.memory, "sim_norm_std", None),
            use_z_score_normalization=config.experiment.llb_use_z_score_normalization,
            dedup_by_task_id=bool(getattr(config.experiment, "llb_dedup_by_task_id", False)),
        )

        # Load snapshot memory state
        memory_service.load_checkpoint_snapshot(str(EXP_DIR), ckpt_id=str(section_num))
        logger.info("Loaded snapshot %d memory state", section_num)

        # Build runner in eval-only mode
        from memrl.run.llb_rl_runner import LLBRLRunner
        runner = LLBRLRunner(
            memory_service=memory_service,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            task=config.experiment.task,
            dataset=val_data,
            valid_dataset=None,
            config=config,
            algorithm=config.experiment.algorithm,
            mode="train",  # must be != "test" for _evaluate to run
            batch_size=config.experiment.batch_size,
            max_steps=config.experiment.max_steps,
            num_sections=1,
            valid_interval=0,
            os_timeout=20,
            ck_dir=EXP_DIR,
        )

        # Run evaluation on val set (inference only, no memory update)
        sr_info = runner._evaluate_single(val_data, "Validation", section_num)
        n_success = sum(1 for r in sr_info if r.get("success"))
        n_total = len(sr_info)
        val_sr = n_success / n_total if n_total > 0 else 0.0

        logger.info("Section %d val SR = %.2f%% (%d/%d)", section_num, val_sr * 100, n_success, n_total)
        results.append({
            "section": section_num,
            "val_sr": round(val_sr * 100, 2),
            "n_success": n_success,
            "n_total": n_total,
        })

        # Cleanup
        try:
            memory_service.cleanup()
        except Exception:
            pass

    # Save summary
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    logger.info("=" * 60)
    logger.info("All done. Results saved to %s", RESULTS_PATH)
    for r in results:
        logger.info("  S%d: val_sr=%.2f%%", r["section"], r["val_sr"])


if __name__ == "__main__":
    main()
