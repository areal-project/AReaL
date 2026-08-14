# memrl/run/alfworld_rl_runner.py
import logging
from pathlib import Path
from typing import Dict, Set, Any
import os

# Kept configurable so a launcher can select a safe ALFWorld no-op without
# changing the behavior of concurrent experiments.
ALFWORLD_FALLBACK_ACTION = (
    os.environ.get("MEMRL_ALFWORLD_FALLBACK_ACTION", "look").strip() or "look"
)
import yaml
import time
import textworld
import textworld.agents
import textworld.gym
import numpy as np
import pandas as pd
import json
import random
from datetime import datetime
from functools import partial

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from tqdm import tqdm
from .base_runner import BaseRunner
from memrl.envs.alfworld_env import AlfWorldEnv
from memrl.agent.memp_agent import MempAgent, INVALID_LLM_ACTION
from memrl.agent.history import EpisodeHistory
from memrl.service.memory_service import MemoryService
from memrl.service.value_driven import RLConfig
from alfworld.agents.environment.alfred_tw_env import (  # type: ignore
    AlfredTWEnv,
    AlfredDemangler,
    AlfredInfos,
    AlfredExpert,
)
MAX_RETRIES = 4
RETRY_DELAY = 2
# Per-batch parallel LLM call cap. Was historically 4 (legacy OpenAI API rate
# limit). For local vLLM with TP=4 on Qwen2.5-72B, 32 is fine and removes the
# ~10x slowdown observed in no-mem eval (where most tasks run to max_steps).
MAX_LLM_CONCURRENCY = max(
    1, int(os.environ.get("MEMRL_ALFWORLD_LLM_CONCURRENCY", "32"))
)
# Invalid HTTP-200 completions are repaired after the batch, not retried inline.
# Inline retries amplify peak gateway load and turn transient transport faults into
# repeated bad actions. These defaults are intentionally conservative and overridable.
DEFERRED_REPAIR_ENABLED = os.environ.get("MEMRL_ALFWORLD_DEFERRED_REPAIR", "1").strip().lower() not in {"0", "false", "no"}
DEFERRED_REPAIR_COOLDOWN_S = max(0.0, float(os.environ.get("MEMRL_ALFWORLD_DEFERRED_REPAIR_COOLDOWN_S", "30")))
DEFERRED_REPAIR_MAX_GAMES = max(0, int(os.environ.get("MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES", "8")))
DEFERRED_REPAIR_ROUNDS = max(1, int(os.environ.get("MEMRL_ALFWORLD_DEFERRED_REPAIR_ROUNDS", "1") or "1"))
try:
    DEFERRED_REPAIR_COOLDOWNS = [
        max(0.0, float(x.strip()))
        for x in os.environ.get("MEMRL_ALFWORLD_DEFERRED_REPAIR_COOLDOWNS", "").split(",")
        if x.strip()
    ]
except ValueError:
    DEFERRED_REPAIR_COOLDOWNS = []

logger = logging.getLogger(__name__)

def load_config_from_path(config_path: str, params=None):
    assert os.path.exists(config_path), f"Invalid config file: {config_path}"
    with open(config_path) as reader:
        config = yaml.safe_load(reader)
    if params is not None:
        for param in params:
            fqn_key, value = param.split("=")
            entry_to_change = config
            keys = fqn_key.split(".")
            for k in keys[:-1]:
                entry_to_change = entry_to_change[k]
            entry_to_change[keys[-1]] = value
    return config

class AlfworldRunner(BaseRunner):
    """
    A Runner that prepares batches of environments for a large-scale experiment.
    It handles loading, splitting the dataset, and creating all necessary
    environment instances upfront.
    """
    def __init__(self, agent: MempAgent, root: str, env_config: str, memory_service: MemoryService, exp_name: str,
                 num_section: int, batch_size: int, max_steps: int, rl_config, ck_dir:str, retrieve_k: int=1, mode: str='train',
                 valid_interval: int=2, test_interval: int=2, dataset_ratio: float=1.0, random_seed: int=42, bon: int=0,
                 ckpt_resume_enabled: bool = False, ckpt_resume_path: Optional[str] = None, ckpt_resume_epoch: Optional[int] = None,
                 baseline_mode: Optional[str] = None, baseline_k: int = 10,
                 holdout_subtask: Optional[str] = None,
                 val_lambda_max: Optional[float] = None,
                 holdout_eval_pools: Optional[List[str]] = None,
                 n_eval_runs: int = 1,
                 eval_temperature: Optional[float] = None,
                 reset_legacy_region_evidence_on_resume: bool = False,
                 region_evidence_sharpen_alpha_override: Optional[float] = None):
        self.agent = agent
        self.root = root
        self.memory_service = memory_service
        self.exp_name = exp_name
        self.random_seed = random_seed
        self.num_section = num_section
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.retrieve_k = retrieve_k
        self.mode = mode
        self.env_config_path = env_config # Store path for AlfWorldEnv wrapper
        self.env_config = load_config_from_path(env_config) # Load config for AlfredTWEnv
        self.valid_interval = valid_interval
        self.test_interval = test_interval
        self.dataset_ratio = dataset_ratio
        self.bon = bon
        self.results_log = []
        self.ckpt_resume_enabled = ckpt_resume_enabled
        self.ckpt_resume_path = ckpt_resume_path
        self.ckpt_resume_epoch = ckpt_resume_epoch
        self.baseline_mode = (baseline_mode or "").strip().lower() or None
        self.baseline_k = max(1, int(baseline_k))
        self.n_eval_runs = max(1, int(n_eval_runs))
        self.eval_temperature = eval_temperature
        self.reset_legacy_region_evidence_on_resume = bool(reset_legacy_region_evidence_on_resume)
        self._legacy_region_evidence_reset_applied = False
        self.region_evidence_sharpen_alpha_override = region_evidence_sharpen_alpha_override
        # Optional short-window experiment cutoff. A value like 40 means the
        # resumed run completes and checkpoints batch 40, then exits before 41.
        try:
            self._stop_after_batch = max(0, int(os.environ.get("MEMRL_ALFWORLD_STOP_AFTER_BATCH", "0") or "0"))
        except (TypeError, ValueError):
            self._stop_after_batch = 0

        # Region failure summary injection (see docs/REGION_FAILURE_SUMMARY.md)
        # Set via configure_failure_summary() after construction.
        self._failure_summary_n_slots = 0
        self._region_failure_summaries = None

        # Single-bank zero-shot holdout: if set (e.g. "alf/pick_and_place_simple"),
        # held-out subtask games are excluded from train and the memory pool.
        # Held-out games from ALL splits (train+valid+test) are combined into
        # holdout_eval_game_files so we can measure zero-shot transfer to the
        # unseen subtask after training on the remaining 6 subtasks.
        self.holdout_subtask: Optional[str] = (holdout_subtask or "").strip() or None
        if self.holdout_subtask and not self.holdout_subtask.startswith("alf/"):
            self.holdout_subtask = f"alf/{self.holdout_subtask}"

        # val_lambda_max: when set (e.g. 0.15), _evaluate temporarily lowers
        # region_manager.shrinkage_lambda_max for the duration of eval so
        # retrieval is region-utility-dominated rather than per-memory-Q-dominated.
        # Mirrors BCB val phase pattern from commit 3c3b76f. Has no effect on
        # non-region runs (no region_manager attribute).
        self.val_lambda_max: Optional[float] = val_lambda_max

        # Holdout eval pool selection: which BCB-style splits to draw the eval
        # bucket from. Default ['valid','test'] = strictest zero-shot (unseen
        # envs). Pass ['train','valid'] for BCB-aligned setup (eval on seen-env
        # held-out games for larger N).
        self.holdout_eval_pools: List[str] = list(holdout_eval_pools) if holdout_eval_pools else ['valid', 'test']

        self.rl_config: Optional[RLConfig] = rl_config


        env_controller = AlfredTWEnv(self.env_config, train_eval="train")
        all_train_game_files = env_controller.game_files

        if not 0.0 < self.dataset_ratio <= 1.0:
            raise ValueError(f"dataset_ratio must be between 0.0 and 1.0, but got {self.dataset_ratio}")

        env_controller = AlfredTWEnv(self.env_config, train_eval='eval_in_distribution')
        self.valid_game_files = env_controller.game_files

        env_controller = AlfredTWEnv(self.env_config, train_eval='eval_out_of_distribution')
        self.test_game_files = env_controller.game_files

        # Holdout filtering must run BEFORE dataset_ratio sampling, otherwise we
        # both shrink train AND drop a subtask, getting an unpredictably small
        # train set. Build the holdout eval bucket here too.
        self.holdout_eval_game_files: List[str] = []
        if self.holdout_subtask:
            from memrl.configs.task_hierarchy import get_primary_subtask

            # Validate spelling against known ALFWorld subtasks (fail fast on typo).
            _KNOWN_ALF_SUBTASKS = {
                "alf/pick_and_place_simple",
                "alf/pick_and_place_with_movable_recep",
                "alf/pick_clean_then_place_in_recep",
                "alf/pick_cool_then_place_in_recep",
                "alf/pick_heat_then_place_in_recep",
                "alf/pick_two_obj_and_place",
                "alf/look_at_obj_in_light",
            }
            if self.holdout_subtask not in _KNOWN_ALF_SUBTASKS:
                raise ValueError(
                    f"Unknown holdout_subtask '{self.holdout_subtask}'. "
                    f"Must be one of: {sorted(_KNOWN_ALF_SUBTASKS)}"
                )

            def _gamefile_subtask(gf: str) -> str:
                parts = gf.split('/')
                if len(parts) < 3:
                    return "alf/unknown"
                task_dir = parts[-3]
                return get_primary_subtask("alfworld", {
                    "task_type": task_dir,
                    "game_file": task_dir,
                })

            # Filter train pool by subtask BEFORE applying dataset_ratio.
            before_train = len(all_train_game_files)
            all_train_game_files = [
                g for g in all_train_game_files
                if _gamefile_subtask(g) != self.holdout_subtask
            ]
            dropped_train = before_train - len(all_train_game_files)

            # Build holdout eval bucket. Source pools controlled by
            # self.holdout_eval_pools (defaults to ['valid','test']).
            # Common choices:
            #   ['valid','test']   — unseen environments only, strictest zero-shot
            #   ['train','valid']  — fuller coverage (includes seen envs), bigger N
            #   ['train','valid','test'] — everything
            requested_pools = list(getattr(self, 'holdout_eval_pools', ['valid', 'test']))
            valid_pool_names = {'train', 'valid', 'test'}
            for p in requested_pools:
                if p not in valid_pool_names:
                    raise ValueError(f"Unknown holdout_eval_pool '{p}'. Must be subset of {sorted(valid_pool_names)}")

            pools_for_holdout_eval: Dict[str, list] = {}
            if 'train' in requested_pools:
                # Use the ORIGINAL train list (pre-filter) for the train-from-holdout bucket
                env_controller = AlfredTWEnv(self.env_config, train_eval="train")
                pools_for_holdout_eval['train'] = env_controller.game_files
            if 'valid' in requested_pools:
                pools_for_holdout_eval['valid'] = self.valid_game_files
            if 'test' in requested_pools:
                pools_for_holdout_eval['test'] = self.test_game_files
            per_split_counts = {}
            seen_paths: set = set()
            for split_name, pool in pools_for_holdout_eval.items():
                holdout_games = [g for g in pool if _gamefile_subtask(g) == self.holdout_subtask]
                # Dedup across splits (defensive — alfworld splits should be disjoint).
                new_games = [g for g in holdout_games if g not in seen_paths]
                seen_paths.update(new_games)
                per_split_counts[split_name] = len(new_games)
                self.holdout_eval_game_files.extend(new_games)

            logger.info(
                "[HOLDOUT] subtask=%s | dropped %d/%d train games | "
                "holdout_eval bucket: %d total (%s)",
                self.holdout_subtask, dropped_train, before_train,
                len(self.holdout_eval_game_files),
                ", ".join(f"{k}={v}" for k, v in per_split_counts.items()),
            )
            if not all_train_game_files:
                raise ValueError(
                    f"After filtering holdout subtask '{self.holdout_subtask}', "
                    f"train pool is empty. Check spelling."
                )
            if not self.holdout_eval_game_files:
                raise ValueError(
                    f"Holdout subtask '{self.holdout_subtask}' produced 0 eval games. "
                    f"Check spelling."
                )

        # Now apply dataset_ratio on the (possibly filtered) train pool.
        if self.dataset_ratio < 1.0:
            num_total_train = len(all_train_game_files)
            num_to_sample = int(num_total_train * self.dataset_ratio)

            logger.info(f"Randomly sampling {num_to_sample} games from the {num_total_train} training games ({self.dataset_ratio:.2%})...")

            # Set a seed for reproducibility of the random sample
            random.seed(self.random_seed)
            self.train_game_files = random.sample(all_train_game_files, k=num_to_sample)
        else:
            # If ratio is 1.0, use the full (filtered) dataset
            logger.info(f"Using the full training set of {len(all_train_game_files)} games.")
            self.train_game_files = all_train_game_files

        # Audit log: clear breakdown of pre-filter → post-filter → post-ratio sizes.
        if self.holdout_subtask:
            logger.info(
                "[HOLDOUT] Final train sizes: pre_filter=%d, post_holdout=%d, post_ratio=%d",
                before_train, len(all_train_game_files), len(self.train_game_files),
            )

        # --- [TENSORBOARD] Initialize SummaryWriter ---
        # Create a unique, timestamped directory for this experiment's logs
        tb_log_dir = self.root / "logs" / "tensorboard" / f"exp_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        # `ck_dir` is the destination for this run.  A resume path is strictly an
        # input source: redirecting `ck_dir` to it makes a continuation write its
        # checkpoints into the parent experiment, and batch-checkpoint cleanup can
        # then delete the continuation's early checkpoints against stale batches.
        # Keep output isolated even when loading a snapshot from another run.
        self.ck_dir = Path(ck_dir)
        self._resume_source_root = (
            Path(self.ckpt_resume_path)
            if self.ckpt_resume_enabled and self.ckpt_resume_path
            else None
        )
        self.local_cache_dir = self.ck_dir / "local_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        # Persist per-game evaluation artifacts alongside the checkpoint state.
        # This path is deliberately independent of the transient TensorBoard
        # directory and survives process restarts/resume in the experiment output.
        self.log_dir = self.local_cache_dir
        self._cum_state_path = self.local_cache_dir / "cum_state.json"
        self._cum_success_ids: Set[str] = set()
        self._cum_total = len(set(self.train_game_files))
        self._resume_section_start, self._resume_batch_start = self._resume_from_ckpt()

    def _analyze_and_report_results(self):
        """
        Analyzes and reports the final results for both training and evaluation,
        including success rates and average steps for all phases.
        """
        if not self.results_log:
            logger.warning("No results were logged. Cannot perform analysis.")
            return

        logger.info("\n" + "#"*20 + " FULL EXPERIMENT FINISHED - FINAL RESULTS " + "#"*20)
        results_df = pd.DataFrame(self.results_log)
        
        # --- Training Performance ---
        train_df = results_df[results_df['mode'].isin(['build', 'update'])]
        if not train_df.empty:
            overall_success_rate = train_df['success'].mean()
            logger.info("\n--- Training Performance (on Train Set) ---")
            logger.info(f"Total Training Trajectories: {len(train_df)}")
            logger.info(f"Overall Success Rate: {overall_success_rate:.2%}")

            section_performance = train_df.groupby('section').agg(
                success_rate=('success', 'mean'),
                avg_steps=('steps', 'mean')
            ).reset_index()
            logger.info("\n>>> Training Performance by Section <<<")
            print(section_performance.to_string(index=False, formatters={'success_rate': '{:.2%}'.format}))
        
        # --- Evaluation Performance ---
        eval_df = results_df[~results_df['mode'].isin(['build', 'update'])]
        if not eval_df.empty:
            logger.info("\n--- Evaluation Performance Summary ---")

            # Pivot table for Success Rate on Eval Sets
            logger.info("\n>>> Success Rate (%) by Evaluation Set <<<")
            # In eval logs, the 'success' column already holds the rate
            eval_success_summary = eval_df.pivot_table(index='after_section', columns='mode', values='success')
            with pd.option_context('display.float_format', '{:.2%}'.format):
                print(eval_success_summary)
            
            # Pivot table for Average Steps on Success on Eval Sets
            logger.info("\n>>> Average Steps on Success by Evaluation Set <<<")
            # In eval logs, the 'steps' column holds the average steps on success
            eval_steps_summary = eval_df.pivot_table(index='after_section', columns='mode', values='steps')
            with pd.option_context('display.float_format', '{:.2f}'.format):
                print(eval_steps_summary)
            
        # --- Save results to a CSV file ---
        log_dir = self.root / "logs"
        log_dir.mkdir(exist_ok=True)
        results_csv_path = log_dir / f"experiment_results_{self.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        results_df.to_csv(results_csv_path, index=False)
        logger.info(f"\nDetailed results saved to: {results_csv_path}")

    def envs_spilt(
        self, 
        game_files, 
        task_type: str
    ) -> List[List[List[str]]]:
        """
        Use the full dataset for each section (no splitting by data). num_section
        only controls how many passes (sections) we run; every section sees the
        entire training set. Batching is done only by mini-batch size.
        """
        logger.info(f"Preparing full dataset batches for task type: '{task_type}'...")
        
        if not game_files:
            raise ValueError(f"No game files found for task_type '{task_type}'. Check your config paths.")

        # Each section uses the whole dataset; num_section controls number of passes.
        section_splits = [game_files for _ in range(self.num_section)]

        game_list_by_section = []
        for i, section_games in enumerate(section_splits):
            section_games = list(section_games)
            
            num_mini_batches = int(np.ceil(len(section_games) / self.batch_size))
            mini_batch_splits = []
            
            for j in range(num_mini_batches):
                start_index = j * self.batch_size
                end_index = start_index + self.batch_size
                mini_batch = section_games[start_index:end_index]
                if mini_batch:
                    mini_batch_splits.append(mini_batch)
            
            game_list_by_section.append(mini_batch_splits)
            logger.info(
                f"Section {i+1}: {len(section_games)} games, split into "
                f"{len(mini_batch_splits)} mini-batches of size <= {self.batch_size}."
            )

        return game_list_by_section
    
    def envs_built(self, mini_batch_games: List[str], task_type: str) -> List[AlfWorldEnv]:
        """
        Receives a 2D list of game files for a SINGLE section and creates a dedicated,
        parallel gym environment for each mini-batch within that section.

        Args:
            section_mini_batches (List[List[str]]): The split game files for one section,
                                                    structured as [mini_batch][game_file].
            task_type (str): The dataset split being used.

        Returns:
            List[AlfWorldEnv]: A 1D list of fully initialized AlfWorldEnv wrappers,
                               where each element is a parallel environment for one mini-batch.
        """
        logger.info(f"Building environment instances for the current section batch...")
        
        # This logic is based on the AlfredTWEnv.init_env source you provided
        domain_randomization = self.env_config["env"]["domain_randomization"]
        if task_type != "train":
            domain_randomization = False

        wrappers = [partial(AlfredDemangler, shuffle=domain_randomization), AlfredInfos]

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        expert_type = self.env_config["env"]["expert_type"]
        training_method = self.env_config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = self.env_config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = self.env_config["dagger"]["training"]["max_nb_steps_per_episode"]
            expert_plan = True if task_type == "train" else False
            if expert_plan:
                wrappers.append(partial(AlfredExpert, expert_type=expert_type))
                request_infos.extras.append("expert_plan")
        else:
            raise NotImplementedError
        
            # The actual batch size for this env is the number of games in its mini-batch
        current_batch_size = len(mini_batch_games)
        env_id = textworld.gym.register_games(
            mini_batch_games, 
            request_infos,
            batch_size=current_batch_size,
            auto_reset=False,
            asynchronous=False,
            max_episode_steps=max_nb_steps_per_episode,
            wrappers=wrappers
        )
        # Launch the underlying Gym environment
        underlying_env = textworld.gym.make(env_id)

        # Wrap it with our AlfWorldEnv for a consistent interface
        env_wrapper = AlfWorldEnv(
            config_path=self.env_config_path,
            preconfigured_env=underlying_env,
            batch_size=current_batch_size
        )
        # Hand env_id to the wrapper so its close() can unregister it from
        # gym's process-global registry (HIGH #2: prevents per-batch leak).
        env_wrapper._textworld_env_id = env_id

        logger.info("Environment instances for this section batch have been built successfully.")
        return env_wrapper

    def _resolve_resume_dir(self) -> Optional[Path]:
        if not self.ckpt_resume_path:
            return None
        base = Path(self.ckpt_resume_path)
        if self.ckpt_resume_epoch is not None:
            if base.name == "snapshot":
                return base / str(self.ckpt_resume_epoch)
            if (base / "snapshot").exists():
                return base / "snapshot" / str(self.ckpt_resume_epoch)
            return base / str(self.ckpt_resume_epoch)

        snapshot_dir = base
        if base.name == "snapshot":
            snapshot_dir = base
        elif (base / "snapshot").exists():
            snapshot_dir = base / "snapshot"

        best = self._find_latest_checkpoint(snapshot_dir)
        if best is not None:
            return best
        return snapshot_dir

    @staticmethod
    def _is_snapshot_complete(snapshot_path: Path) -> bool:
        """Check if a snapshot directory has the minimum required files.

        Accepts either MemoryService format (cube/textual_memory.json) or
        Mem0MemoryService format (mem0_qdrant/ directory).
        """
        # MemoryService format
        cube_dir = snapshot_path / "cube"
        if cube_dir.is_dir():
            textual_mem = cube_dir / "textual_memory.json"
            if textual_mem.is_file() and textual_mem.stat().st_size >= 10:
                return True

        # Mem0MemoryService format
        mem0_qdrant = snapshot_path / "mem0_qdrant"
        if mem0_qdrant.is_dir() and any(mem0_qdrant.iterdir()):
            return True

        return False

    @staticmethod
    def _find_latest_checkpoint(snapshot_dir: Path) -> Optional[Path]:
        """Find the latest complete checkpoint (section or batch) in the snapshot directory."""
        import re
        if not snapshot_dir.exists():
            return None
        batch_re = re.compile(r"^s(\d+)_b(\d+)$")
        section_candidates = []
        batch_candidates = []
        for d in snapshot_dir.iterdir():
            if not d.is_dir():
                continue
            if not AlfworldRunner._is_snapshot_complete(d):
                logger.warning("Skipping incomplete snapshot: %s", d.name)
                continue
            if d.name.isdigit():
                section_candidates.append((int(d.name), d))
            else:
                m = batch_re.match(d.name)
                if m:
                    batch_candidates.append((int(m.group(1)), int(m.group(2)), d))
        best_section_int = -1
        if section_candidates:
            section_candidates.sort(key=lambda x: x[0])
            best_section_int = section_candidates[-1][0]
        best_batch = (-1, -1)
        best_batch_path = None
        if batch_candidates:
            batch_candidates.sort(key=lambda x: (x[0], x[1]))
            best_batch = (batch_candidates[-1][0], batch_candidates[-1][1])
            best_batch_path = batch_candidates[-1][2]
        if best_batch_path and best_batch[0] >= best_section_int:
            return best_batch_path
        if best_section_int >= 0:
            return section_candidates[-1][1]
        return None

    def _load_cum_state(self, path: Optional[Path] = None) -> None:
        path = path or self._cum_state_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ids = payload.get("success_ids", [])
            self._cum_success_ids = {str(x) for x in ids if x}
            total = payload.get("total")
            if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                self._cum_total = total
            global_step = payload.get("global_step")
            if isinstance(global_step, int) and not isinstance(global_step, bool) and global_step >= 0:
                self._global_step = global_step
                logger.info(f"Restored _global_step={global_step} from {path}")
            initial_eval_completed = payload.get("initial_eval_completed")
            if isinstance(initial_eval_completed, bool):
                self._initial_eval_completed = initial_eval_completed
            last_topology_section = payload.get("last_region_topology_section")
            if isinstance(last_topology_section, int) and not isinstance(last_topology_section, bool):
                self._last_region_topology_section = last_topology_section
            # Holdout sanity check on resume: persisted vs current must match,
            # otherwise we'd silently mix experiments (e.g. resumed without
            # --holdout_subtask flag → all 7 subtasks now in train, invalidating
            # the experiment). Fail fast unless the user explicitly opts out.
            persisted_holdout = payload.get("holdout_subtask")
            if persisted_holdout is not None and getattr(self, "holdout_subtask", None) != persisted_holdout:
                raise ValueError(
                    f"Resume holdout mismatch: persisted holdout_subtask={persisted_holdout!r} "
                    f"but current run was started with holdout_subtask={self.holdout_subtask!r}. "
                    f"Use the same --holdout_subtask flag (or none) as the original run, "
                    f"or delete cum_state.json to force a fresh start."
                )
        except Exception:
            logger.warning("Failed to load cumulative acc state from %s", path, exc_info=True)

    def _persist_cum_state(self, path: Optional[Path] = None) -> None:
        path = path or self._cum_state_path
        payload = {
            "success_ids": sorted(self._cum_success_ids),
            "total": self._cum_total,
            "global_step": getattr(self, "_global_step", 0),
            "initial_eval_completed": getattr(self, "_initial_eval_completed", False),
            "last_region_topology_section": getattr(self, "_last_region_topology_section", -1),
            "holdout_subtask": getattr(self, "holdout_subtask", None),
            "updated_at": datetime.now().isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: write to temp file then rename. Avoids partially-written JSON
            # on crash mid-write, which would break resume next session.
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            logger.warning("Failed to persist cumulative acc state to %s", path, exc_info=True)

    def _cleanup_old_batch_checkpoints(self, section_num: int, current_batch: int, max_keep: int = 3) -> None:
        """Remove old batch-level checkpoints, keeping only the latest `max_keep`."""
        import shutil as _shutil
        snapshot_dir = Path(self.ck_dir) / "snapshot"
        if not snapshot_dir.exists():
            return
        prefix = f"s{section_num}_b"
        batch_dirs = sorted(
            [d for d in snapshot_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
            key=lambda d: int(d.name[len(prefix):]) if d.name[len(prefix):].isdigit() else 0,
        )
        to_remove = batch_dirs[:-max_keep] if len(batch_dirs) > max_keep else []
        for old_dir in to_remove:
            try:
                _shutil.rmtree(old_dir)
                logger.info(f"Removed old batch checkpoint: {old_dir.name}")
            except Exception:
                logger.warning(f"Failed to remove old checkpoint {old_dir.name}", exc_info=True)

    def _update_cum_success(self, trajectories: List[Dict[str, Any]]) -> None:
        for traj in trajectories:
            key = (
                traj.get("gamefile")
                or traj.get("task_id")
                or traj.get("task_description")
            )
            if not key:
                continue
            if traj.get("success"):
                self._cum_success_ids.add(str(key))

    def _current_cum_acc(self) -> float:
        if self._cum_total <= 0:
            return 0.0
        return len(self._cum_success_ids) / self._cum_total

    def _sanitize_reflection_trajectory(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        task_start_prefix = "Now, it's your turn to solve a new task."
        last_task_start_idx = None

        for idx, msg in enumerate(trajectory or []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role != "user":
                continue
            if not isinstance(content, str):
                content = str(content)
            if content.strip().startswith(task_start_prefix):
                last_task_start_idx = idx

        if last_task_start_idx is not None:
            for msg in (trajectory or [])[last_task_start_idx + 1 :]:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    continue
                if not isinstance(content, str):
                    content = str(content)
                if content.strip().startswith("You attempted this task before."):
                    continue
                if content.strip().startswith("Here is an example of how to solve the task:"):
                    continue
                if role == "user" and content.strip().startswith("Observation:"):
                    cleaned.append({"role": "user", "content": content})
                elif role == "assistant":
                    cleaned.append({"role": "assistant", "content": content})
            return cleaned

        # Fallback when task marker is missing: keep only obs/action, drop example header.
        for msg in trajectory or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                continue
            if not isinstance(content, str):
                content = str(content)
            if content.strip().startswith("You attempted this task before."):
                continue
            if content.strip().startswith("Here is an example of how to solve the task:"):
                continue
            if role == "user" and content.strip().startswith("Observation:"):
                cleaned.append({"role": "user", "content": content})
            elif role == "assistant":
                cleaned.append({"role": "assistant", "content": content})
        return cleaned

    def _format_reflection_note(self, trajectory: List[Dict[str, Any]], success: bool) -> str:
        status = "CORRECT" if success else "INCORRECT"
        sanitized = self._sanitize_reflection_trajectory(trajectory)
        try:
            traj_text = json.dumps(sanitized, ensure_ascii=False, default=str)
        except Exception:
            traj_text = str(sanitized)
        return (
            "You attempted this task before.\n"
            f"Result: {status}\n"
            "Previous trajectory (observations/actions only):\n"
            f"{traj_text}\n\n"
            "Reflect on mistakes or improvements and solve the task again with a better plan."
        )

    def _sample_from_batch_baseline(
        self,
        mini_batch_env: AlfWorldEnv,
        *,
        reflection_notes: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        completed_experiences = []
        current_bs = mini_batch_env.batch_size
        active_slots = list(range(current_bs))
        messages_per_slot: List[List[Dict]] = [[] for _ in range(current_bs)]
        steps_per_slot: List[int] = [0 for _ in range(current_bs)]

        results = mini_batch_env.reset()
        current_task_descs = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        current_observations = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        task_types = ['/'.join(res['info']['extra.gamefile'].split('/')[-3:-1]) for res in results]
        current_gamefiles = [
            res.get('info', {}).get('extra.gamefile') or res.get('info', {}).get('gamefile')
            for res in results
        ]

        for i in range(current_bs):
            messages = self.agent._construct_messages(
                task_description=current_task_descs[i],
                retrieved_memories={},
                task_type=task_types[i]
            )
            if reflection_notes:
                note = reflection_notes.get(current_gamefiles[i])
                if note:
                    messages.insert(-1, {"role": "system", "content": note})
            messages_per_slot[i] = messages

        for step in tqdm(range(self.max_steps), desc="Sampling mini-batch (baseline)"):
            if not active_slots:
                logger.info("All active tasks finished. Ending batch early.")
                break
            slots_to_act_on = active_slots

            actions_dict = {}
            new_messages_dict: Dict[int, List[Dict[str, str]]] = {}
            with ThreadPoolExecutor(max_workers=min(len(slots_to_act_on), MAX_LLM_CONCURRENCY)) as executor:
                future_to_slot = {}
                for i in slots_to_act_on:
                    def submit_with_retry(slot_idx=i):
                        history_snapshot = list(messages_per_slot[slot_idx])
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                return self.agent.act(
                                    observation=current_observations[slot_idx],
                                    history_messages=history_snapshot,
                                    first_step=(step == 0)
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[Sampling Retry] Slot {slot_idx} attempt {attempt}/{MAX_RETRIES} failed: {e}"
                                )
                                if attempt < MAX_RETRIES:
                                    time.sleep(RETRY_DELAY)
                                else:
                                    logger.error(f"[Sampling Abort] Slot {slot_idx} all retries failed.")
                                    return ("inventory", [])
                    future_to_slot[executor.submit(submit_with_retry)] = i

                for future in as_completed(future_to_slot):
                    slot_idx = future_to_slot[future]
                    try:
                        result = future.result()
                        if isinstance(result, tuple) and len(result) == 2:
                            actions_dict[slot_idx], new_messages_dict[slot_idx] = result
                        else:
                            actions_dict[slot_idx] = result
                            new_messages_dict[slot_idx] = []
                    except Exception as e:
                        logger.error(f"[Sampling Fatal] Slot {slot_idx} raised unhandled exception: {e}")
                        actions_dict[slot_idx] = "inventory"
                        new_messages_dict[slot_idx] = []

            # Apply history updates serially (no thread race).
            for slot_idx, new_msgs in new_messages_dict.items():
                if new_msgs:
                    messages_per_slot[slot_idx].extend(new_msgs)

            for slot_idx in slots_to_act_on:
                steps_per_slot[slot_idx] += 1
            actions = [ALFWORLD_FALLBACK_ACTION] * current_bs
            for slot_idx, action in actions_dict.items():
                actions[slot_idx] = action

            valid_actions = []
            for i, act in enumerate(actions):
                if act is None:
                    valid_actions.append(ALFWORLD_FALLBACK_ACTION)
                elif not isinstance(act, str) or not act.strip():
                    valid_actions.append(ALFWORLD_FALLBACK_ACTION)
                else:
                    valid_actions.append(act)
            actions = valid_actions

            step_results = mini_batch_env.step(actions)

            newly_finished_slots = []
            for i in active_slots:
                result = step_results[i]
                current_observations[i] = result['obs']
                info = result.get("info", {}) or {}
                gamefile = info.get("extra.gamefile") or info.get("gamefile")
                if gamefile:
                    current_gamefiles[i] = gamefile

                aborted = bool(info.get('error'))
                if result['done'] or aborted:
                    success = (not aborted) and (result.get('reward', 0) > 0)
                    if aborted:
                        logger.warning(f"Slot {i} aborted due to env error: {info.get('error')}")
                    completed_experiences.append({
                        "task_description": current_task_descs[i],
                        "trajectory": list(messages_per_slot[i]),
                        "success": success,
                        "steps": steps_per_slot[i],
                        "gamefile": current_gamefiles[i],
                        "task_type": task_types[i],
                        "aborted": aborted,
                        "timeout": False,
                    })
                    newly_finished_slots.append(i)

            if newly_finished_slots:
                active_slots = [s for s in active_slots if s not in newly_finished_slots]

        # Handle incomplete trajectories — slots that never hit done within max_steps.
        # Same logic as _sample_from_batch: active_slots is the complement of
        # completed_experiences by construction, so no dedup needed.
        for i in active_slots:
            if not messages_per_slot[i]:
                continue
            completed_experiences.append({
                "task_description": current_task_descs[i],
                "trajectory": list(messages_per_slot[i]),
                "success": False,
                "steps": steps_per_slot[i],
                "gamefile": current_gamefiles[i],
                "task_type": task_types[i],
                "aborted": False,
                "timeout": True,
            })

        return completed_experiences

    def _run_passk_baseline(self) -> None:
        total_tasks = len(self.train_game_files)
        solved: Set[str] = set()
        summary = []
        result_path = self.local_cache_dir / "baseline_passk_results.jsonl"
        summary_path = self.local_cache_dir / "baseline_passk_summary.json"
        state_path = self.local_cache_dir / "baseline_passk_state.json"

        # Each round = 1 pass over the full train set (1 attempt per game).
        # baseline_k rounds = pass@k.
        all_games = list(self.train_game_files)
        num_batches = int(np.ceil(len(all_games) / self.batch_size))
        batches = [all_games[i * self.batch_size:(i + 1) * self.batch_size] for i in range(num_batches)]

        start_round = 1
        start_batch = 0
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                solved = {str(x) for x in state.get("solved", [])}
                summary = state.get("summary", [])
                last_round = int(state.get("last_round", 0))
                last_batch = int(state.get("last_batch", -1))
                if state.get("round_complete", False):
                    start_round = last_round + 1
                    start_batch = 0
                else:
                    start_round = last_round
                    start_batch = last_batch + 1
                logger.info(
                    "Resuming pass@k from round %d batch %d (%d solved)",
                    start_round, start_batch, len(solved),
                )
            except Exception:
                logger.warning("Failed to load pass@k state from %s", state_path, exc_info=True)

        if start_round > self.baseline_k:
            logger.info("pass@k already completed (last round %d).", start_round - 1)
            return

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting pass@k round %d/%d", round_idx, self.baseline_k)
            for batch_idx, mini_batch_games in enumerate(tqdm(batches, desc=f"pass@k round {round_idx}")):
                if round_idx == start_round and batch_idx < start_batch:
                    continue
                pending_games = [g for g in mini_batch_games if g not in solved]
                if not pending_games:
                    continue
                mini_batch_env = self.envs_built(pending_games, 'train')
                trajectories = self._sample_from_batch_baseline(mini_batch_env)
                mini_batch_env.close()
                for traj in trajectories:
                    if traj.get("success"):
                        key = traj.get("gamefile") or traj.get("task_description")
                        if key:
                            solved.add(str(key))
                    payload = {
                        "round": round_idx,
                        "baseline": "passk",
                        **traj,
                    }
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "last_round": round_idx,
                        "last_batch": batch_idx,
                        "round_complete": False,
                        "solved": sorted(solved),
                        "summary": summary,
                    }, f, ensure_ascii=False)
            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            logger.info("pass@k round %d cumulative acc: %.2f%%", round_idx, cum_acc * 100)
            self.writer.add_scalar("Baseline/PassK_Cumulative_Acc", cum_acc, round_idx)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "last_round": round_idx,
                    "last_batch": batch_idx,
                    "round_complete": True,
                    "solved": sorted(solved),
                    "summary": summary,
                }, f, ensure_ascii=False)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _run_reflection_baseline(self) -> None:
        total_tasks = len(self.train_game_files)
        solved: Set[str] = set()
        summary = []
        result_path = self.local_cache_dir / "baseline_reflection_results.jsonl"
        summary_path = self.local_cache_dir / "baseline_reflection_summary.json"
        state_path = self.local_cache_dir / "baseline_reflection_state.json"
        train_sections_data = self.envs_spilt(self.train_game_files, 'train')
        reflection_notes: Dict[str, str] = {}

        start_round = 1
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                solved = {str(x) for x in state.get("solved", [])}
                reflection_notes = {
                    str(k): v for k, v in state.get("reflection_notes", {}).items()
                }
                last_completed = int(state.get("last_completed_round", 0))
                start_round = max(1, last_completed + 1)
                logger.info("Resuming reflection baseline from round %d", start_round)
            except Exception:
                logger.warning("Failed to load reflection baseline state from %s", state_path, exc_info=True)

        if start_round > self.baseline_k:
            logger.info("Reflection baseline already completed (last round %d).", start_round - 1)
            return

        for round_idx in range(start_round, self.baseline_k + 1):
            logger.info("Starting reflection round %d/%d", round_idx, self.baseline_k)
            for section_data in train_sections_data:
                for mini_batch_games in tqdm(section_data, desc=f"reflection round {round_idx}"):
                    pending_games = [g for g in mini_batch_games if g not in solved]
                    if not pending_games:
                        continue
                    mini_batch_env = self.envs_built(pending_games, 'train')
                    trajectories = self._sample_from_batch_baseline(
                        mini_batch_env,
                        reflection_notes=reflection_notes,
                    )
                    mini_batch_env.close()
                    for traj in trajectories:
                        key = traj.get("gamefile") or traj.get("task_description")
                        if key:
                            key = str(key)
                            reflection_notes[key] = self._format_reflection_note(
                                traj.get("trajectory", []),
                                bool(traj.get("success")),
                            )
                        if traj.get("success") and key:
                            solved.add(str(key))
                        payload = {
                            "round": round_idx,
                            "baseline": "reflection",
                            **traj,
                        }
                        with open(result_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            cum_acc = (len(solved) / total_tasks) if total_tasks > 0 else 0.0
            summary.append({"round": round_idx, "cum_acc": cum_acc, "solved": len(solved), "total": total_tasks})
            logger.info("reflection round %d cumulative acc: %.2f%%", round_idx, cum_acc * 100)
            self.writer.add_scalar("Baseline/Reflection_Cumulative_Acc", cum_acc, round_idx)
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "last_completed_round": round_idx,
                            "solved": sorted(solved),
                            "reflection_notes": reflection_notes,
                            "total": total_tasks,
                            "updated_at": datetime.now().isoformat(),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                logger.warning("Failed to save reflection baseline state to %s", state_path, exc_info=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _resume_from_ckpt(self) -> tuple:
        """Returns (start_section, start_batch) for resuming training."""
        import re
        if not self.ckpt_resume_enabled:
            # Auto-resume fallback: if ck_dir already has snapshots (e.g. platform
            # retry with same MEMRL_RUN_ID), resume from latest checkpoint.
            auto_snapshot_root = Path(self.ck_dir) / "snapshot"
            if auto_snapshot_root.exists():
                best = self._find_latest_checkpoint(auto_snapshot_root)
                if best is not None:
                    logger.info("[Auto-resume] Found existing snapshot in ck_dir, resuming from %s", best)
                    return self._load_and_resolve_resume(best)
            return (1, 1)
        # A continuation may specify an external source checkpoint for its first
        # launch.  Once this run has written any complete checkpoint of its own,
        # prefer that newest local snapshot on future platform retries.  Otherwise
        # every retry would replay from the original source and discard all work
        # done by the continuation.
        own_snapshot_root = Path(self.ck_dir) / "snapshot"
        own_best = self._find_latest_checkpoint(own_snapshot_root)
        if own_best is not None:
            logger.info(
                "[Auto-resume] Resume source configured, but found newer local "
                "continuation snapshot; resuming from %s", own_best,
            )
            return self._load_and_resolve_resume(own_best)

        snapshot_dir = self._resolve_resume_dir()
        if not snapshot_dir or not snapshot_dir.exists():
            logger.warning("Resume enabled but snapshot not found: %s", snapshot_dir)
            return (1, 1)

        return self._load_and_resolve_resume(snapshot_dir)

    def _load_and_resolve_resume(self, snapshot_dir: Path) -> tuple:
        """Load memory snapshot + cum_state and return (start_section, start_batch)."""
        import re
        loaded = False
        loaded_checkpoint_id: Optional[int] = None
        if hasattr(self.memory_service, "load_checkpoint_snapshot"):
            try:
                loaded_checkpoint_id = self.memory_service.load_checkpoint_snapshot(
                    str(snapshot_dir), local_cache_dir=str(self.local_cache_dir)
                )
                loaded = True
            except TypeError:
                try:
                    loaded_checkpoint_id = self.memory_service.load_checkpoint_snapshot(str(snapshot_dir))
                    loaded = True
                except Exception:
                    logger.warning("Failed to load checkpoint snapshot from %s", snapshot_dir, exc_info=True)
            except Exception:
                logger.warning("Failed to load checkpoint snapshot from %s", snapshot_dir, exc_info=True)

        if not loaded:
            return (1, 1)

        rm = getattr(self.memory_service, "region_manager", None)
        if self.region_evidence_sharpen_alpha_override is not None and rm is not None:
            rm.region_evidence_sharpen_alpha = max(0.1, float(self.region_evidence_sharpen_alpha_override))
            logger.info("Applied resumed region evidence sharpen alpha override: %.3f", rm.region_evidence_sharpen_alpha)

        retemperature = os.environ.get("MEMRL_REGION_RETEMPERATURE", "").strip()
        if retemperature and rm is not None:
            rm.retemper_memberships_and_reroute_source_evidence(float(retemperature))
            logger.info("Applied resumed Region membership temperature override: %s", retemperature)

        if self.reset_legacy_region_evidence_on_resume and not self._legacy_region_evidence_reset_applied:
            reset = getattr(rm, "reset_legacy_observed_evidence", None)
            if callable(reset):
                reset(reason=f"clean resume from {snapshot_dir}")
                self._legacy_region_evidence_reset_applied = True
            else:
                logger.warning("Requested legacy region-evidence reset but RegionManager lacks the migration method")

        state_candidates = []
        if isinstance(loaded_checkpoint_id, int) and loaded_checkpoint_id > 0:
            if snapshot_dir.name.isdigit() and int(snapshot_dir.name) == loaded_checkpoint_id:
                state_candidates.append(snapshot_dir / "local_cache" / "cum_state.json")
            else:
                state_candidates.append(snapshot_dir / str(loaded_checkpoint_id) / "local_cache" / "cum_state.json")
        state_candidates.append(snapshot_dir / "local_cache" / "cum_state.json")
        state_candidates.append(self._cum_state_path)

        loaded_state = False
        for state_path in state_candidates:
            if state_path.exists():
                self._load_cum_state(state_path)
                loaded_state = True
                if state_path == self._cum_state_path:
                    logger.warning(
                        "Resumed with FALLBACK cum_state (instance-level path, not snapshot-local). "
                        "Potentially inconsistent (snapshot, cum_state) pair — replayed batches may "
                        "skip global_step / success_ids updates. Snapshot dir: %s",
                        snapshot_dir,
                    )
                break
        if not loaded_state:
            self._load_cum_state(snapshot_dir / "local_cache" / "cum_state.json")

        batch_match = re.match(r"^s(\d+)_b(\d+)$", snapshot_dir.name)
        if batch_match:
            section = int(batch_match.group(1))
            batch = int(batch_match.group(2))
            logger.info("Resuming from batch checkpoint %s → section %d, batch %d", snapshot_dir.name, section, batch + 1)
            return (section, batch + 1)

        resume_epoch = self.ckpt_resume_epoch
        if resume_epoch is None:
            # A numeric snapshot directory (e.g. snapshot/8) is the durable
            # checkpoint section ID. Some backends return another positive int
            # from load_checkpoint_snapshot (Mem0 returns memory count), which
            # must never override the path-derived epoch.
            try:
                resume_epoch = int(snapshot_dir.name)
            except ValueError:
                resume_epoch = (
                    loaded_checkpoint_id
                    if isinstance(loaded_checkpoint_id, int) and loaded_checkpoint_id > 0
                    else None
                )
        start_section = (resume_epoch + 1) if resume_epoch else 1
        return (start_section, 1)

    def configure_failure_summary(self, n_slots: int = 2, summaries_path: Optional[str] = None,
                                   replace_with_summary: bool = True, mode: str = "region",
                                   inline_k: Optional[int] = None,
                                   force_recall: bool = False):
        """Enable failure summary injection.

        Args:
            n_slots: number of top-k slots reserved for failure memories (default 2 of 5)
            summaries_path: path to JSON file from build_region_failure_summaries.py.
            replace_with_summary: if True (default), replace failure content with summary.
            mode: "region" (default) uses per-region aggregated summary;
                  "inline" aggregates the retrieved top-k failures on-the-fly (ablation).
            inline_k: for mode=inline, how many failures to aggregate. None = use all.
            force_recall: reserve failure slots using failure-only recall when True;
                otherwise replace only failures naturally present in baseline top-K.
        """
        import json as _json
        self._failure_summary_n_slots = n_slots
        self._failure_summary_force_recall = bool(force_recall)
        self._failure_summary_replace = replace_with_summary
        self._failure_summary_mode = mode
        self._failure_summary_inline_k = inline_k
        if summaries_path and replace_with_summary and mode == "region":
            data = _json.loads(Path(summaries_path).read_text())
            self._region_failure_summaries = data.get("summaries", {})
            logger.info(
                "[Region Failure Summary] loaded %d region summaries from %s, n_slots=%d, replace=True",
                len(self._region_failure_summaries), summaries_path, n_slots,
            )
        else:
            self._region_failure_summaries = None
            logger.info(
                "[Failure Summary] n_slots=%d, mode=%s, inline_k=%s, replace=%s, force_recall=%s",
                n_slots, mode, inline_k, replace_with_summary, bool(force_recall),
            )

    def configure_success_summary(self, n_slots: int = 0, mode: str = "append"):
        """Enable region success pattern summary injection (symmetric to failure summary).

        Args:
            n_slots: number of slots for region success summary (0 = disabled).
            mode: "append" = add success summary as extra slot(s) on top of raw
                  success scripts. "replace" = replace n_slots raw success memories
                  with region success summary (fully symmetric to failure).
        """
        self._success_summary_n_slots = n_slots
        self._success_summary_mode = mode
        logger.info(
            "[Region Success Summary] n_slots=%d, mode=%s", n_slots, mode,
        )

    def process_retrieve_mems(self, retrieved_mems_per_slot, task_descs_per_slot=None):
        n_failure_slots = getattr(self, '_failure_summary_n_slots', 0)

        # For inline mode, we may need more failures for aggregation than n_failure_slots
        _inline_mode = getattr(self, '_failure_summary_mode', 'region') == 'inline'
        _inline_k = getattr(self, '_failure_summary_inline_k', None)
        _effective_failure_k = max(n_failure_slots, _inline_k or 0) if _inline_mode else n_failure_slots

        force_recall = bool(getattr(self, '_failure_summary_force_recall', False))
        extra_failure_per_slot = [None] * len(retrieved_mems_per_slot)
        if force_recall and n_failure_slots > 0 and task_descs_per_slot:
            from concurrent.futures import ThreadPoolExecutor
            slots_needing_failure = []
            for i, mems_for_one_slot in enumerate(retrieved_mems_per_slot):
                n_existing_fail = sum(
                    1 for m in mems_for_one_slot
                    if not m['metadata'].model_extra.get('success', False)
                )
                if n_existing_fail < _effective_failure_k:
                    extra_needed = _effective_failure_k - n_existing_fail
                    exclude_ids = {m.get('memory_id') for m in mems_for_one_slot}
                    slots_needing_failure.append((i, task_descs_per_slot[i], extra_needed, exclude_ids))
            if slots_needing_failure:
                with ThreadPoolExecutor(max_workers=min(len(slots_needing_failure), 8)) as executor:
                    futures = [
                        (idx, executor.submit(self._retrieve_failure_only, desc, k=k, exclude_ids=exc))
                        for idx, desc, k, exc in slots_needing_failure
                    ]
                    for idx, fut in futures:
                        try:
                            extra_failure_per_slot[idx] = fut.result()
                        except Exception:
                            extra_failure_per_slot[idx] = []

        processed_mems_per_slot = []
        for i, mems_for_one_slot in enumerate(retrieved_mems_per_slot):
            success_mems = []
            failed_mems = []

            for mem in mems_for_one_slot:
                is_success = mem['metadata'].model_extra.get('success', False)
                if is_success:
                    success_mems.append(mem)
                else:
                    failed_mems.append(mem)

            # --- Region failure replacement: conditional or forced recall ---
            if n_failure_slots > 0:
                if force_recall:
                    if extra_failure_per_slot[i] is not None:
                        failed_mems.extend(extra_failure_per_slot[i])
                    success_mems = success_mems[:max(0, self.retrieve_k - n_failure_slots)]
                    failed_mems = failed_mems[:_effective_failure_k]
                    total = len(success_mems) + len(failed_mems)
                    if total < self.retrieve_k:
                        selected_ids = {m.get('memory_id') for m in success_mems + failed_mems}
                        remaining = [m for m in mems_for_one_slot if m.get('memory_id') not in selected_ids]
                        success_mems.extend(remaining[:self.retrieve_k - total])
                eligible_failures = failed_mems[:min(n_failure_slots, len(failed_mems))]
                if eligible_failures and getattr(self, '_failure_summary_replace', True):
                    mode = getattr(self, '_failure_summary_mode', 'region')
                    if mode == 'inline':
                        self._replace_failure_with_inline_summary(eligible_failures)
                    elif mode == 'global':
                        self._replace_failure_with_global_summary(eligible_failures)
                    else:
                        self._replace_failure_with_region_summary(eligible_failures)
            # Conditional preserves all-success top-K; forced reserves failure slots.

            # --- Region success summary injection (symmetric to failure) ---
            n_success_slots = getattr(self, '_success_summary_n_slots', 0)
            if n_success_slots > 0:
                mode = getattr(self, '_success_summary_mode', 'append')
                ss_mems = self._make_success_summary_mems(
                    mems_for_one_slot, n_success_slots,
                )
                if mode == 'replace' and ss_mems:
                    # Drop n_success_slots raw success mems, prepend summaries
                    success_mems = ss_mems + success_mems[len(ss_mems):]
                elif ss_mems:  # append mode
                    success_mems = ss_mems + success_mems

            final_mems = {}
            if success_mems:
                final_mems['successed'] = success_mems
            if failed_mems:
                final_mems['failed'] = failed_mems

            processed_mems_per_slot.append(final_mems)

        # --- Failure summary injection stats (batch-level log) ---
        if getattr(self, '_failure_summary_n_slots', 0) > 0:
            n_success_total = sum(len(s.get('successed', [])) for s in processed_mems_per_slot if isinstance(s, dict))
            n_failure_total = sum(len(s.get('failed', [])) for s in processed_mems_per_slot if isinstance(s, dict))
            n_replaced = sum(
                1 for s in processed_mems_per_slot if isinstance(s, dict)
                for fm in s.get('failed', [])
                if isinstance(fm, dict) and fm.get('_region_failure_summary')
            )
            if not hasattr(self, '_failure_inject_log_counter'):
                self._failure_inject_log_counter = 0
            self._failure_inject_log_counter += 1
            # Log every 10 batches + first batch
            if self._failure_inject_log_counter <= 2 or self._failure_inject_log_counter % 10 == 0:
                logger.info(
                    "[Failure Summary Stats] batch #%d: %d success, %d failure injected "
                    "(%d replaced with region summary) across %d slots",
                    self._failure_inject_log_counter, n_success_total, n_failure_total,
                    n_replaced, len(processed_mems_per_slot),
                )
                # Print sample prompt content for first failure with summary
                if n_replaced > 0:
                    for s in processed_mems_per_slot:
                        if not isinstance(s, dict):
                            continue
                        for fm in s.get('failed', []):
                            if isinstance(fm, dict) and fm.get('_region_failure_summary'):
                                content = fm.get('content', '')
                                logger.info(
                                    "[Failure Summary Sample] (first 300 chars): %s",
                                    content[:300],
                                )
                                break
                        else:
                            continue
                        break

        return processed_mems_per_slot

    def _selfrag_critique_batch(self, retrieved_mems_per_slot, task_descs):
        """Self-RAG: LLM critique filters irrelevant memories for each slot."""
        import re as _re
        filtered = []
        futures = []
        with ThreadPoolExecutor(max_workers=min(len(task_descs), MAX_LLM_CONCURRENCY)) as executor:
            for i, (mems_dict, desc) in enumerate(zip(retrieved_mems_per_slot, task_descs)):
                if not isinstance(mems_dict, dict):
                    futures.append((i, None))
                    continue
                all_mems = mems_dict.get('successed', []) + mems_dict.get('failed', [])
                if not all_mems:
                    futures.append((i, None))
                    continue
                futures.append((i, executor.submit(self._selfrag_critique_single, desc, all_mems)))

        results = [None] * len(task_descs)
        total_before = 0
        total_after = 0
        for i, fut in futures:
            mems_dict = retrieved_mems_per_slot[i]
            if fut is None or not isinstance(mems_dict, dict):
                results[i] = mems_dict
                continue
            try:
                kept = fut.result(timeout=60)
            except Exception as e:
                logger.warning("[Self-RAG] critique slot %d failed: %s, keeping all", i, e)
                results[i] = mems_dict
                all_mems = mems_dict.get('successed', []) + mems_dict.get('failed', [])
                total_before += len(all_mems)
                total_after += len(all_mems)
                continue
            all_mems = mems_dict.get('successed', []) + mems_dict.get('failed', [])
            total_before += len(all_mems)
            total_after += len(kept)
            kept_set = set(id(m) for m in kept)
            results[i] = {
                'successed': [m for m in mems_dict.get('successed', []) if id(m) in kept_set],
                'failed': [m for m in mems_dict.get('failed', []) if id(m) in kept_set],
            }
        logger.info("[Self-RAG] Critique kept %d/%d memories across %d slots", total_after, total_before, len(task_descs))
        return results

    def _selfrag_critique_single(self, task_description: str, memories: List[Dict]) -> List[Dict]:
        """Critique a single slot's memories, return only relevant ones."""
        import re as _re
        numbered = []
        for i, m in enumerate(memories):
            content = m.get('content', '') or ''
            if not content:
                meta = m.get('metadata', None)
                if meta:
                    content = (getattr(meta, 'model_extra', {}) or {}).get('full_content', '') or ''
            numbered.append(f"[Memory {i+1}]\n{content[:1500]}")

        critique_prompt = (
            "You are a relevance judge for a household task agent. "
            "Given the current task and retrieved memories from past attempts, "
            "decide which memories are RELEVANT and could help solve the current task.\n\n"
            f"Current task: {task_description[:500]}\n\n"
            "Retrieved memories:\n" + "\n\n".join(numbered) + "\n\n"
            "Return ONLY a JSON list of relevant memory numbers (1-indexed). "
            "If none are relevant, return []\n"
            "Example: [1, 3]"
        )
        resp = self.agent.llm.generate(
            messages=[{"role": "user", "content": critique_prompt}],
            temperature=0.0,
            max_tokens=128,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        match = _re.search(r'\[[\d\s,]*\]', resp or "")
        if not match:
            logger.warning("[Self-RAG] malformed critique response; keeping all")
            return memories
        indices = json.loads(match.group())
        return [memories[idx - 1] for idx in indices if 1 <= idx <= len(memories)]

    def _retrieve_failure_only(self, task_description: str, k: int = 2,
                               exclude_ids: Optional[set] = None) -> List[Dict]:
        """Retrieve top-k failure memories by sim only (no Q rerank).

        Uses a pre-built failure index (_failure_query_keys) for O(F) scan
        instead of O(N) full dict_memory scan. Index is rebuilt lazily when
        dict_memory grows.
        """
        if not hasattr(self.memory_service, 'dict_memory') or not self.memory_service.dict_memory:
            return []
        if not hasattr(self.memory_service, '_mem_cache'):
            return []

        try:
            from memrl.service.memory_service import get_embedding_with_retry
            embed = getattr(self.memory_service.embedding_provider, 'embed', None)
            if not callable(embed):
                return []

            # Get query embedding (use cache if available)
            _qe = getattr(self.memory_service, 'query_embeddings', {})
            query_vec = _qe.get(task_description)
            if query_vec is None:
                query_vec = get_embedding_with_retry(embed, [task_description])[0]

            import math
            query_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-8

            # --- Lazily build/rebuild failure index ---
            # Maps query_key → [failure_mem_ids] for fast lookup
            mc = self.memory_service._mem_cache
            dict_mem = self.memory_service.dict_memory
            current_size = len(dict_mem)

            if (not hasattr(self, '_failure_index') or
                    self._failure_index is None or
                    getattr(self, '_failure_index_size', 0) != current_size):
                # Rebuild index
                self._failure_index = {}  # query_key → [failure_mem_id, ...]
                for query_key, mem_ids in dict_mem.items():
                    fail_ids = []
                    for mid in mem_ids:
                        mem_obj = mc.get(mid)
                        if mem_obj is None:
                            continue
                        md = getattr(mem_obj, 'metadata', {})
                        if hasattr(md, 'model_extra'):
                            is_success = md.model_extra.get('success', True)
                        elif isinstance(md, dict):
                            is_success = md.get('success', True)
                        else:
                            continue
                        if not is_success:
                            fail_ids.append(mid)
                    if fail_ids:
                        self._failure_index[query_key] = fail_ids
                self._failure_index_size = current_size

            # --- Score failure memories by sim (only scan failure index) ---
            candidates = []
            for query_key, fail_ids in self._failure_index.items():
                qv = _qe.get(query_key)
                if qv is None:
                    continue
                q_norm = math.sqrt(sum(x * x for x in qv)) or 1e-8
                sim = sum(a * b for a, b in zip(query_vec, qv)) / (query_norm * q_norm)
                for mid in fail_ids:
                    if exclude_ids and mid in exclude_ids:
                        continue
                    mem_obj = mc.get(mid)
                    if mem_obj is None:
                        continue
                    md = getattr(mem_obj, 'metadata', {})
                    content = None
                    try:
                        if hasattr(md, 'model_extra'):
                            content = md.model_extra.get('full_content')
                        elif isinstance(md, dict):
                            content = md.get('full_content')
                    except Exception:
                        pass
                    candidates.append({
                        'memory_id': mid,
                        'content': content,
                        'similarity': float(sim),
                        'metadata': mem_obj,
                    })

            # Sort by sim, take top-k
            candidates.sort(key=lambda c: c['similarity'], reverse=True)
            return candidates[:k]
        except Exception as e:
            logger.warning("[failure-only retrieve] failed: %s", e)
            return []

    def _ensure_failure_index_built(self):
        """Pre-build the failure index so parallel retrieval threads don't race on it."""
        if not hasattr(self.memory_service, 'dict_memory') or not self.memory_service.dict_memory:
            return
        if not hasattr(self.memory_service, '_mem_cache'):
            return
        mc = self.memory_service._mem_cache
        dict_mem = self.memory_service.dict_memory
        current_size = len(dict_mem)
        if (not hasattr(self, '_failure_index') or
                self._failure_index is None or
                getattr(self, '_failure_index_size', 0) != current_size):
            self._failure_index = {}
            for query_key, mem_ids in dict_mem.items():
                fail_ids = []
                for mid in mem_ids:
                    mem_obj = mc.get(mid)
                    if mem_obj is None:
                        continue
                    md = getattr(mem_obj, 'metadata', {})
                    if hasattr(md, 'model_extra'):
                        is_success = md.model_extra.get('success', True)
                    elif isinstance(md, dict):
                        is_success = md.get('success', True)
                    else:
                        continue
                    if not is_success:
                        fail_ids.append(mid)
                if fail_ids:
                    self._failure_index[query_key] = fail_ids
            self._failure_index_size = current_size
            n_fail = sum(len(v) for v in self._failure_index.values())
            logger.info("[failure_index] pre-built: %d query_keys with failures, %d total failure mems",
                        len(self._failure_index), n_fail)


    def _prewarm_query_embeddings(self):
        """Batch-embed all memory query keys that are not yet cached.

        Without this, each parallel retrieve_query thread hits the embedding API
        serially for missing keys (line 1394 in memory_service.py). By pre-warming
        here, the threads find all keys cached and skip that path entirely.
        """
        ms = self.memory_service
        if not hasattr(ms, 'dict_memory') or not ms.dict_memory:
            return
        embed_fn = getattr(getattr(ms, 'embedding_provider', None), 'embed', None)
        if not callable(embed_fn):
            return
        _qe = getattr(ms, 'query_embeddings', None)
        if _qe is None:
            return

        queries = list(ms.dict_memory.keys())
        missing = [q for q in queries if q not in _qe]
        if not missing:
            logger.info("[PREWARM] all %d query key embeddings already cached", len(queries))
            return

        logger.info("[PREWARM] computing embeddings for %d/%d missing query keys...", len(missing), len(queries))
        try:
            # The service owns a single-flight lock shared with retrieve_query.
            # This prevents the first parallel training batch from duplicating
            # the same cold-cache fill if prewarm overlaps a retrieval.
            ensure = getattr(ms, "_ensure_query_key_embeddings", None)
            if callable(ensure):
                ensure(queries, batch_size=64)
            else:  # compatibility for external MemoryService implementations
                from memrl.service.memory_service import get_embedding_with_retry
                for i in range(0, len(missing), 64):
                    chunk = missing[i:i+64]
                    for q, v in zip(chunk, get_embedding_with_retry(embed_fn, chunk)):
                        _qe[q] = v
            logger.info("[PREWARM] done, %d embeddings computed", len(missing))
        except Exception as e:
            logger.warning("[PREWARM] failed: %s", e)

    @staticmethod
    def _argmax_region_for_memory(rm, mem_id):
        """Return the memory's highest-weight region, tolerating old snapshots."""
        if rm is None or not getattr(rm, 'regions', None) or not mem_id:
            return None
        weights = getattr(rm, 'membership_weights', {}).get(mem_id)
        if weights is None:
            return None
        values = np.asarray(weights, dtype=float).reshape(-1)
        n = min(len(values), len(rm.regions))
        if n == 0 or not np.isfinite(values[:n]).any():
            return None
        safe_values = np.where(np.isfinite(values[:n]), values[:n], -np.inf)
        return rm.regions[int(np.argmax(safe_values))]

    def _replace_failure_with_region_summary(self, failed_mems: List[Dict]) -> None:
        """Replace failure memory content with its argmax region's summary.

        Reads directly from region.failure_summary (built dynamically after
        clustering by RegionManager._build_region_failure_summaries).
        Falls back to external JSON summaries (global) if region has no summary.
        """
        rm = getattr(self.memory_service, 'region_manager', None) if hasattr(self, 'memory_service') else None

        # External fallback (global summary from JSON, if loaded)
        external_summaries = getattr(self, '_region_failure_summaries', None)
        global_summary = external_summaries.get('global', '') if external_summaries else ''

        for fm in failed_mems:
            mem_id = fm.get('memory_id')
            region = self._argmax_region_for_memory(rm, mem_id)
            # Prefer live region.failure_summary (built after clustering)
            summary = region.failure_summary if region and region.failure_summary else ''
            # Fallback: external global summary
            if not summary:
                summary = global_summary
            if summary:
                fm['content'] = summary
                fm['_region_failure_summary'] = True

    def _replace_failure_with_inline_summary(self, failed_mems: List[Dict]) -> None:
        """Aggregate retrieved failure mems into inline summaries (ablation).

        Splits the top-k retrieved failures into two tiers and generates a
        separate summary for each, filling up to n_failure_slots. Tier 1 covers
        the most similar failures (rank 1..k/2), Tier 2 the next tier
        (rank k/2+1..k). This preserves the 2-slot structure while giving the
        agent both a focused and a broader failure perspective.
        """
        from memrl.service.region_manager import RegionManager

        k = getattr(self, '_failure_summary_inline_k', None)
        mems_to_aggregate = failed_mems[:k] if k else failed_mems

        if not mems_to_aggregate:
            return

        # Split into two tiers
        mid = max(1, len(mems_to_aggregate) // 2)
        tier1 = mems_to_aggregate[:mid]
        tier2 = mems_to_aggregate[mid:]

        def _build_summary(mems, label):
            fields_list = []
            for fm in mems:
                content = fm.get('content', '')
                if not content:
                    continue
                fields = RegionManager._parse_failure_fields(content)
                if fields["failure_mode"] or fields["mistakes"]:
                    fields_list.append(fields)
            if not fields_list:
                return None
            return RegionManager._format_failure_summary(fields_list, top_n=3)

        summary1 = _build_summary(tier1, "top")
        summary2 = _build_summary(tier2, "secondary")

        if summary1 and summary2 and len(failed_mems) >= 2:
            failed_mems[0]['content'] = summary1
            failed_mems[0]['_region_failure_summary'] = True
            failed_mems[1]['content'] = summary2
            failed_mems[1]['_region_failure_summary'] = True
            del failed_mems[2:]
        elif summary1:
            failed_mems[0]['content'] = summary1
            failed_mems[0]['_region_failure_summary'] = True
            del failed_mems[1:]

    def _replace_failure_with_global_summary(self, failed_mems: List[Dict]) -> None:
        """Aggregate ALL failure memories in the store into one global summary (ablation).

        Treats the entire memory store as one big region — no region clustering needed.
        This tests whether region-level grouping adds value over a single global aggregate.
        The summary is cached and rebuilt every ~3000 steps to amortize cost.
        """
        from memrl.service.region_manager import RegionManager

        cache_key = getattr(self, '_global_step', 0) // 3000
        cached = getattr(self, '_global_fs_cache', None)
        if cached and cached[0] == cache_key:
            summary = cached[1]
        else:
            # Use the same lookup interface as region FS
            ms = self.memory_service
            mem_cache = getattr(ms, '_mem_cache', None)
            if not mem_cache:
                return

            # Try to use the callback if available (RegionMemoryService)
            lookup_fn = getattr(ms, '_get_mem_content_and_success', None)

            fields_list = []
            for mem_id in list(mem_cache.keys()):
                try:
                    if lookup_fn:
                        content, success = lookup_fn(mem_id)
                    else:
                        # Fallback: try direct Pydantic access
                        mem_obj = mem_cache[mem_id]
                        md = getattr(mem_obj, 'metadata', None)
                        if md and hasattr(md, 'model_extra'):
                            content = md.model_extra.get('full_content', '')
                            success = md.model_extra.get('success', True)
                        else:
                            continue
                except Exception:
                    continue
                if success is not False:
                    continue
                if not content:
                    continue
                fields = RegionManager._parse_failure_fields(content)
                if fields["failure_mode"] or fields["mistakes"]:
                    fields_list.append(fields)

            if not fields_list:
                return

            summary = RegionManager._format_failure_summary(fields_list, top_n=3)
            self._global_fs_cache = (cache_key, summary)
            logger.info(
                "[Global Failure Summary] built from %d failure memories (cache_key=%d)",
                len(fields_list), cache_key,
            )

        if not summary or not failed_mems:
            return

        failed_mems[0]['content'] = summary
        failed_mems[0]['_region_failure_summary'] = True
        del failed_mems[1:]

    def _make_success_summary_mems(self, mems_for_one_slot, n_slots: int) -> List[Dict]:
        """Build up to n_slots region-success-summary pseudo-memories for a slot.

        Finds the region(s) of the slot's retrieved memories and returns their
        success_summary as memory dicts (marked _region_success_summary) so the
        agent prompt renders them in the SUCCESSFUL MEMORIES section.
        """
        rm = getattr(self.memory_service, 'region_manager', None) if hasattr(self, 'memory_service') else None
        if not rm or not rm.regions:
            return []

        # Collect distinct argmax-region success summaries from this slot's mems,
        # ordered by appearance (most similar first).
        seen_regions = set()
        out = []
        for m in mems_for_one_slot:
            mid = m.get('memory_id')
            region = self._argmax_region_for_memory(rm, mid)
            if region is None or id(region) in seen_regions:
                continue
            summary = getattr(region, 'success_summary', '')
            if not summary:
                continue
            seen_regions.add(id(region))
            out.append({
                'memory_id': f'success_summary_r{region.region_id}',
                'content': summary,
                'similarity': 1.0,
                '_region_success_summary': True,
            })
            if len(out) >= n_slots:
                break
        return out

    @staticmethod
    def _lightweight_mems(mems_for_slot):
        """Strip heavy `metadata` (Pydantic model) and `memory_item` refs from
        a slot's retrieved mems, keeping only the small fields needed downstream
        (memory_id, content, similarity, success bool).

        Without this, every trajectory in section_trajectories pins entire
        Pydantic metadata trees + memory_item objects until section end,
        causing OOM on the no-region (MemRL) path where retrieval doesn't
        narrow the candidate pool via region q_cache swap.
        """
        if not isinstance(mems_for_slot, dict):
            return mems_for_slot
        out = {}
        for bucket_key in ('successed', 'failed'):
            bucket = mems_for_slot.get(bucket_key)
            if not bucket:
                continue
            slim = []
            for m in bucket:
                if not isinstance(m, dict):
                    slim.append(m)
                    continue
                # Extract success flag from metadata (the only field downstream needs)
                md = m.get('metadata')
                success = False
                try:
                    if md is not None and hasattr(md, 'model_extra'):
                        success = bool(md.model_extra.get('success', False))
                    elif isinstance(md, dict):
                        success = bool(md.get('success', False))
                except Exception:
                    pass
                slim.append({
                    'memory_id': m.get('memory_id'),
                    'content': m.get('content'),
                    'similarity': m.get('similarity'),
                    'success': success,
                })
            if slim:
                out[bucket_key] = slim
        return out

    def _register_subtask_embeddings(self) -> None:
        """Register ALFWorld subtask embeddings for zero-shot transfer.

        Uses natural language descriptions of each task type as embeddings.
        """
        import numpy as np

        rm = getattr(self.memory_service, 'region_manager', None)
        if rm is None:
            return

        embed_fn = getattr(getattr(self.memory_service, 'embedding_provider', None), 'embed', None)
        if not callable(embed_fn):
            return

        subtask_descriptions = {
            "alf/look_at_obj_in_light": "Find an object and examine it under a light source like a desk lamp",
            "alf/pick_and_place_simple": "Pick up an object from one location and place it at another location",
            "alf/pick_and_place_with_movable_recep": "Pick up an object, place it in a movable container, then move the container to a target location",
            "alf/pick_clean_then_place_in_recep": "Pick up a dirty object, clean it at a sink, then place it at a target location",
            "alf/pick_cool_then_place_in_recep": "Pick up an object, cool it in a fridge, then place it at a target location",
            "alf/pick_heat_then_place_in_recep": "Pick up an object, heat it in a microwave, then place it at a target location",
            "alf/pick_two_obj_and_place": "Find and pick up two instances of the same object type and place them at a target location",
        }

        try:
            texts = list(subtask_descriptions.values())
            embeddings = embed_fn(texts)
            for (subtask, _), emb in zip(subtask_descriptions.items(), embeddings):
                rm.set_subtask_embedding(subtask, np.array(emb))
            logger.info("Registered %d ALFWorld subtask embeddings for zero-shot transfer", len(subtask_descriptions))
        except Exception as e:
            logger.warning("Failed to register subtask embeddings: %s", e)

    def _sample_from_batch(
        self, mini_batch_env: AlfWorldEnv, *,
        _allow_deferred_repair: bool = True,
        _repair_task_type: str = "train",
    ) -> List[Dict]:
        """
        Runs one parallel environment (a mini-batch), managing the full conversational
        history (messages list) for each parallel game and feeding it to the ReAct agent.
        """
        completed_experiences = []
        current_bs = mini_batch_env.batch_size
        active_slots = list(range(current_bs))
        messages_per_slot: List[List[Dict]] = [[] for _ in range(current_bs)]
        steps_per_slot: List[int] = [0 for _ in range(current_bs)]
        # Slots that received an HTTP-200 response without a usable ALFWorld action.
        # The batched env still receives a no-op, but the complete game is later
        # replayed after cooldown and replaces this contaminated trajectory.
        invalid_response_slots: Set[int] = set()

        results = mini_batch_env.reset()
        current_task_descs = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        # The first observation is part of the initial prompt, not a separate step
        current_observations = ['\n'.join(res['obs'].split('\n\n')[1:]) for res in results]
        # task_types holds "<task_type>/<trial_id>" (e.g. pick_and_place_simple/trial_T20190907_*).
        # Downstream code that needs just the task type (no trial) should index [0] after
        # splitting on "/". We keep the compound form here for trajectory provenance.
        task_types = ['/'.join(res['info']['extra.gamefile'].split('/')[-3:-1]) for res in results]
        # Bare task type (no trial id) for region/subtask lookups.
        task_type_only = [tt.split('/', 1)[0] if '/' in tt else tt for tt in task_types]
        current_gamefiles = [
            res.get('info', {}).get('extra.gamefile') or res.get('info', {}).get('gamefile')
            for res in results
        ]

        # --- Retrieve initial memories for the batch ---
        logger.info(f"Retrieving initial memories (k={self.retrieve_k}) for the batch in parallel...")

        # Compute target_subtasks for region gating
        from memrl.configs.task_hierarchy import get_primary_subtask
        target_subtasks_batch = []
        for idx in range(current_bs):
            game_file = current_gamefiles[idx] or ""
            target_subtasks_batch.append(
                get_primary_subtask("alfworld", {"task_type": task_type_only[idx], "game_file": game_file})
            )

        # --- Batch pre-compute embeddings for all task descriptions ---
        # This avoids 32 threads each hitting the embedding API sequentially.
        _qe = getattr(self.memory_service, 'query_embeddings', None)
        if _qe is not None:
            embed_fn = getattr(getattr(self.memory_service, 'embedding_provider', None), 'embed', None)
            if callable(embed_fn):
                uncached_descs = [d for d in current_task_descs if d not in _qe]
                if uncached_descs:
                    try:
                        from memrl.service.memory_service import get_embedding_with_retry
                        vecs = get_embedding_with_retry(embed_fn, uncached_descs)
                        for desc, vec in zip(uncached_descs, vecs):
                            _qe[desc] = vec
                        logger.info("[BATCH EMBED] pre-computed %d task embeddings", len(uncached_descs))
                    except Exception as e:
                        logger.warning("[BATCH EMBED] failed, falling back to per-thread: %s", e)

        MAX_RETRIEVE_CONCURRENCY = 8  # match embedding server capacity
        with ThreadPoolExecutor(max_workers=min(current_bs, MAX_RETRIEVE_CONCURRENCY)) as executor:
            # target_subtask is region-specific. Base MemoryService.retrieve_query
            # doesn't accept it, so only pass it when the underlying service
            # actually has region_manager (i.e. is RegionMemoryService).
            # Without this guard, baseline runs fail every batch with
            # TypeError: ... unexpected keyword argument 'target_subtask'.
            _supports_target_subtask = getattr(self.memory_service, 'region_manager', None) is not None
            future_re_mems = []
            for idx, desc in enumerate(current_task_descs):
                kw = {
                    'k': self.retrieve_k,
                    'threshold': getattr(self.rl_config, "sim_threshold", getattr(self.rl_config, "tau", 0.0)),
                }
                if _supports_target_subtask:
                    kw['target_subtask'] = target_subtasks_batch[idx]
                future_re_mems.append(
                    executor.submit(self.memory_service.retrieve_query, desc, **kw)
                )

            retrieved_mems_per_slot = []
            retrieved_queries_per_slot = []

            for future in future_re_mems:
                result = future.result()
                if isinstance(result, tuple):
                    ret_dict, topk_queries = result
                    mem = ret_dict.get('selected', []) if isinstance(ret_dict, dict) else []
                else:
                    mem, topk_queries = [], []

                retrieved_mems_per_slot.append(mem)
                retrieved_queries_per_slot.append(topk_queries)

        retrieved_mems_per_slot = self.process_retrieve_mems(
            retrieved_mems_per_slot, task_descs_per_slot=current_task_descs,
        )
        # Sanity log: total memories retrieved in this batch. For no-mem
        # baseline (k_retrieve=0 or empty dict_memory) this should always be 0.
        _total_retrieved = sum(
            (len(s.get('successed', [])) + len(s.get('failed', [])))
            for s in retrieved_mems_per_slot if isinstance(s, dict)
        )
        logger.info(f"[RETRIEVE SANITY] batch retrieved memories: {_total_retrieved} (k={self.retrieve_k}, slots={len(retrieved_mems_per_slot)})")

        if getattr(self, '_selfrag_enabled', False):
            retrieved_mems_per_slot = self._selfrag_critique_batch(retrieved_mems_per_slot, current_task_descs)

        logger.info("Constructing initial ReAct prompts for each game...")
        for i in range(current_bs):
            messages_per_slot[i] = self.agent._construct_messages(
                task_description=current_task_descs[i],
                retrieved_memories=retrieved_mems_per_slot[i],
                task_type=task_types[i]
            )

        for step in tqdm(range(self.max_steps), desc="Sampling mini-batch (ReAct)"):
            if not active_slots:
                logger.info("All active tasks finished. Ending batch early.")
                break
            # --- Determine which slots need an action ---
            slots_to_act_on = active_slots

            actions_dict = {}
            new_messages_dict: Dict[int, List[Dict[str, str]]] = {}

            with ThreadPoolExecutor(max_workers=min(len(slots_to_act_on), MAX_LLM_CONCURRENCY)) as executor:
                future_to_slot = {}

                for i in slots_to_act_on:
                    def submit_with_retry(slot_idx=i):
                        # Snapshot history under the GIL so concurrent retries on
                        # different slots can't observe a half-mutated list.
                        history_snapshot = list(messages_per_slot[slot_idx])
                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                return self.agent.act(
                                    observation=current_observations[slot_idx],
                                    history_messages=history_snapshot,
                                    first_step=(step == 0)
                                )
                            except Exception as e:
                                logger.warning(
                                    f"[Sampling Retry] Slot {slot_idx} attempt {attempt}/{MAX_RETRIES} failed: {e}"
                                )
                                if attempt < MAX_RETRIES:
                                    time.sleep(RETRY_DELAY)
                                else:
                                    logger.error(f"[Sampling Abort] Slot {slot_idx} all retries failed.")
                                    return ("inventory", [])

                    future_to_slot[executor.submit(submit_with_retry)] = i

                for future in as_completed(future_to_slot):
                    slot_idx = future_to_slot[future]
                    try:
                        result = future.result()
                        if isinstance(result, tuple) and len(result) == 2:
                            actions_dict[slot_idx], new_messages_dict[slot_idx] = result
                        else:
                            # Legacy callers / unexpected return — be defensive.
                            actions_dict[slot_idx] = result
                            new_messages_dict[slot_idx] = []
                    except Exception as e:
                        logger.error(f"[Sampling Fatal] Slot {slot_idx} raised unhandled exception: {e}")
                        actions_dict[slot_idx] = "inventory"
                        new_messages_dict[slot_idx] = []

            # A short/unparseable HTTP-200 completion is not an agent action. Do not
            # retry under peak load; mark the game for delayed replay after the batch.
            # The immediate no-op only keeps the batched TextWorld API aligned.
            for slot_idx, action in list(actions_dict.items()):
                if action == INVALID_LLM_ACTION:
                    invalid_response_slots.add(slot_idx)
                    actions_dict[slot_idx] = ALFWORLD_FALLBACK_ACTION
                    logger.warning(
                        "[Deferred Repair] slot=%d received invalid LLM completion; "
                        "will replay game after batch cooldown.", slot_idx,
                    )

            # Apply history updates serially in the main thread (no race).
            for slot_idx, new_msgs in new_messages_dict.items():
                if new_msgs:
                    messages_per_slot[slot_idx].extend(new_msgs)

            # Increment step counter ONLY for slots that actually acted.
            for slot_idx in slots_to_act_on:
                steps_per_slot[slot_idx] += 1

            # Build action vector: only active slots get a real action. Inactive
            # (already-done) slots get a no-op marker that the env wrapper will
            # ignore. We still need to pass `current_bs` actions because
            # textworld's batched env expects an action per slot.
            actions = [ALFWORLD_FALLBACK_ACTION] * current_bs
            for slot_idx, action in actions_dict.items():
                actions[slot_idx] = action

            valid_actions = []
            for i, act in enumerate(actions):
                if act is None:
                    logger.warning(f"[Sampling Warning] Slot {i} action is None, replaced with {ALFWORLD_FALLBACK_ACTION!r}.")
                    valid_actions.append(ALFWORLD_FALLBACK_ACTION)
                elif not isinstance(act, str) or not act.strip():
                    logger.warning(f"[Sampling Warning] Slot {i} invalid action '{act}', replaced with {ALFWORLD_FALLBACK_ACTION!r}.")
                    valid_actions.append(ALFWORLD_FALLBACK_ACTION)
                else:
                    valid_actions.append(act)
            actions = valid_actions

            if step < 3:
                for si in active_slots[:3]:
                    logger.info(f"[DEBUG] Step {step} Slot {si} action: {repr(actions[si])}")

            step_results = mini_batch_env.step(actions)
            # --- Result processing and state update ---

            newly_finished_slots = []

            for i in active_slots:
                result = step_results[i]
                # MED #5: ignore env updates for slots that were already inactive
                # (they shouldn't appear here since we iterate active_slots, but
                # be defensive about backends that mutate done-slot state).
                current_observations[i] = result['obs']
                info = result.get("info", {}) or {}
                gamefile = info.get("extra.gamefile") or info.get("gamefile")
                if gamefile:
                    current_gamefiles[i] = gamefile

                if step < 3:
                    logger.info(f"[DEBUG] Step {step} Slot {i} obs: {repr(result['obs'][:200])} done={result['done']} reward={result.get('reward', 0)}")

                # HIGH #3: env.step failures now mark error=True on the step_data
                # (see AlfWorldEnv.step). Treat aborted episodes as done=True,
                # success=False so the trajectory doesn't fake-extend with empty obs.
                aborted = bool(info.get('error'))
                if result['done'] or aborted:
                    success = (not aborted) and (result.get('reward', 0) > 0)
                    if aborted:
                        logger.warning(f"Slot {i} aborted due to env error: {info.get('error')}")
                    else:
                        logger.info(f"Slot {i} finished a game. Success: {success}")

                    completed_experiences.append({
                        "task_description": current_task_descs[i],
                        "trajectory": list(messages_per_slot[i]),  # snapshot to avoid downstream aliasing
                        "success": success,
                        "retrieved_queries": retrieved_queries_per_slot[i],
                        "retrieved_mems": self._lightweight_mems(retrieved_mems_per_slot[i]),
                        "steps": steps_per_slot[i],
                        "gamefile": current_gamefiles[i],
                        "task_type": task_types[i],
                        "aborted": aborted,
                        "timeout": False,
                    })

                    newly_finished_slots.append(i)
            if newly_finished_slots:
                active_slots = [s for s in active_slots if s not in newly_finished_slots]

        # Handle incomplete trajectories — slots that never hit done within max_steps.
        # Since we remove from active_slots whenever we append to completed_experiences,
        # anything still in active_slots is guaranteed not in completed_experiences;
        # no content-based dedup needed (was O(N²) deep list compare with brittle
        # false-dedup risk when two slots share task text and early trajectory).
        for i in active_slots:
            if not messages_per_slot[i]:
                continue
            completed_experiences.append({
                "task_description": current_task_descs[i],
                "trajectory": list(messages_per_slot[i]),
                "success": False,
                "retrieved_queries": retrieved_queries_per_slot[i],
                "retrieved_mems": self._lightweight_mems(retrieved_mems_per_slot[i]),
                "steps": steps_per_slot[i],
                "gamefile": current_gamefiles[i],
                "task_type": task_types[i],
                # Schema consistency: always set both flags. timeout distinguishes
                # max_steps-exhausted from env-error abort, both being non-success.
                "aborted": False,
                "timeout": True,
            })

        if _allow_deferred_repair and DEFERRED_REPAIR_ENABLED and invalid_response_slots:
            contaminated_games = {
                str(current_gamefiles[i] or '') for i in invalid_response_slots
                if current_gamefiles[i]
            }
            repair_slots = sorted(invalid_response_slots)[:DEFERRED_REPAIR_MAX_GAMES]
            logger.warning(
                "[Deferred Repair] batch has %d contaminated slot(s); replaying %d "
                "complete game(s) serially after %.1fs cooldown.",
                len(invalid_response_slots), len(repair_slots), DEFERRED_REPAIR_COOLDOWN_S,
            )
            repaired_by_gamefile = {}
            for ordinal, slot_idx in enumerate(repair_slots, 1):
                gamefile = current_gamefiles[slot_idx]
                if not gamefile:
                    continue
                for repair_round in range(1, DEFERRED_REPAIR_ROUNDS + 1):
                    cooldown = (
                        DEFERRED_REPAIR_COOLDOWNS[min(repair_round - 1, len(DEFERRED_REPAIR_COOLDOWNS) - 1)]
                        if DEFERRED_REPAIR_COOLDOWNS else DEFERRED_REPAIR_COOLDOWN_S
                    )
                    if cooldown:
                        time.sleep(cooldown)
                    repair_env = None
                    try:
                        logger.info(
                            "[Deferred Repair] replay %d/%d slot=%d round=%d/%d cooldown=%.1fs split=%s game=%s",
                            ordinal, len(repair_slots), slot_idx, repair_round,
                            DEFERRED_REPAIR_ROUNDS, cooldown, _repair_task_type, gamefile,
                        )
                        repair_env = self.envs_built([gamefile], _repair_task_type)
                        replay = self._sample_from_batch(
                            repair_env,
                            _allow_deferred_repair=False,
                            _repair_task_type=_repair_task_type,
                        )
                        if len(replay) == 1 and not replay[0].get('transport_invalid', False):
                            replay[0]['deferred_repair'] = True
                            replay[0]['deferred_repair_round'] = repair_round
                            repaired_by_gamefile[str(gamefile)] = replay[0]
                            logger.info(
                                "[Deferred Repair] replay completed slot=%d round=%d success=%s",
                                slot_idx, repair_round, replay[0].get('success'),
                            )
                            break
                        logger.warning(
                            "[Deferred Repair] replay unusable for slot=%d round=%d/%d",
                            slot_idx, repair_round, DEFERRED_REPAIR_ROUNDS,
                        )
                    except Exception:
                        logger.warning(
                            "[Deferred Repair] replay failed for slot=%d round=%d/%d",
                            slot_idx, repair_round, DEFERRED_REPAIR_ROUNDS,
                            exc_info=True,
                        )
                    finally:
                        if repair_env is not None:
                            try:
                                repair_env.close()
                            except Exception:
                                logger.debug("[Deferred Repair] failed to close replay env", exc_info=True)
            # Replace contaminated originals so fake `look` transitions never enter
            # memory writes, Region evidence, or the reported train SR. If no replay
            # is valid, retain an explicit aborted transport-invalid record for audit.
            final = []
            seen_contaminated = set()
            for traj in completed_experiences:
                key = str(traj.get('gamefile') or '')
                if key in repaired_by_gamefile:
                    final.append(repaired_by_gamefile.pop(key))
                    seen_contaminated.add(key)
                elif key in contaminated_games:
                    traj['transport_invalid'] = True
                    traj['aborted'] = True
                    traj['success'] = False
                    final.append(traj)
                    seen_contaminated.add(key)
                    logger.warning("[Deferred Repair] no valid replay for %s; excluding from learning.", key)
                else:
                    final.append(traj)
            # A contaminated slot can still be active at max_steps with no prior
            # completed record. Preserve it as an explicitly excluded audit entry.
            for key in contaminated_games - seen_contaminated:
                final.append({
                    "task_description": "",
                    "trajectory": [],
                    "success": False,
                    "retrieved_queries": [],
                    "retrieved_mems": {},
                    "steps": 0,
                    "gamefile": key,
                    "task_type": "",
                    "aborted": True,
                    "timeout": False,
                    "transport_invalid": True,
                })
                logger.warning("[Deferred Repair] missing contaminated result for %s; excluding from learning.", key)
            completed_experiences = final
        elif invalid_response_slots:
            # Nested replay observed another invalid completion. Surface it to the
            # parent caller instead of silently accepting its fallback `look` trace.
            contaminated_games = {
                str(current_gamefiles[i] or '') for i in invalid_response_slots
                if current_gamefiles[i]
            }
            for traj in completed_experiences:
                if str(traj.get('gamefile') or '') in contaminated_games:
                    traj['transport_invalid'] = True
                    traj['aborted'] = True
                    traj['success'] = False
                    logger.warning("[Deferred Repair] nested replay remained transport-invalid for %s.", traj.get('gamefile'))

        return completed_experiences

    def _evaluate(self, game_files: List[str], eval_type: str, after_section: int) -> float:
        """
        Runs the agent on a given dataset split for evaluation purposes only.
        No memory building or updating occurs.

        When self.n_eval_runs > 1 AND this is the last section, runs:
          - 1x at temperature 0.0 (greedy, deterministic)
          - (n_eval_runs-1)x at eval_temperature (default 0.2) for CI
        Otherwise runs 1x at original temperature.
        """
        is_last_section = (after_section >= self.num_section)
        if self.n_eval_runs > 1 and is_last_section:
            return self._evaluate_multi(game_files, eval_type, after_section)
        return self._evaluate_single(game_files, eval_type, after_section)

    def _evaluate_multi(self, game_files: List[str], eval_type: str, after_section: int) -> float:
        """Run evaluation n_eval_runs times on the last section.

        Run 0: temperature=0.0 (greedy baseline)
        Run 1..n-1: temperature=eval_temperature (for variance/CI)
        """
        import scipy.stats as st
        rates = []
        eval_temp = self.eval_temperature if self.eval_temperature is not None else 0.2
        for run_idx in range(self.n_eval_runs):
            if run_idx == 0:
                temp_override = 0.0
            else:
                temp_override = eval_temp
            logger.info(f"[{eval_type}] Eval run {run_idx+1}/{self.n_eval_runs} (temp={temp_override})")
            sr = self._evaluate_single(
                game_files, eval_type, after_section,
                log_summary=True, run_idx=run_idx,
                temperature_override=temp_override,
            )
            if sr is not None:
                rates.append(sr)
        if not rates:
            return 0.0
        mean_sr = np.mean(rates)
        if len(rates) >= 2:
            ci = st.t.interval(0.95, len(rates)-1, loc=mean_sr, scale=st.sem(rates))
            ci_half = (ci[1] - ci[0]) / 2
        else:
            ci_half = 0.0
        logger.info(
            f"--- Multi-Eval Summary on {eval_type} (after Section {after_section}) ---\n"
            f"  Runs: {len(rates)}, Mean SR: {mean_sr:.2%} ± {ci_half:.2%} (95%% CI)\n"
            f"  Individual: {[f'{r:.2%}' for r in rates]}"
        )
        self.writer.add_scalar(f"Evaluation/Success_Rate_Mean/{eval_type}", mean_sr, after_section)
        self.writer.add_scalar(f"Evaluation/Success_Rate_CI/{eval_type}", ci_half, after_section)
        return mean_sr

    def _evaluate_single(self, game_files: List[str], eval_type: str, after_section: int, log_summary: bool = True, run_idx: int = 0, temperature_override: Optional[float] = None) -> float:
        """
        Runs the agent on a given dataset split for evaluation purposes only.
        No memory building or updating occurs.

        If `val_lambda_max` was set on the runner, temporarily switch
        `region_manager.shrinkage_lambda_max` for the duration of the eval so
        retrieval at eval time is more region-dominated (consistent with BCB's
        val phase behavior — train uses high lambda for per-memory Q learning,
        eval uses low lambda for region utility transfer).

        Args:
            game_files (list): list of game files
            eval_type (str): A string identifier for logging ('Validation' or 'Test').
            after_setcion (int): num of current section in the train loop
        Returns:
            float: The success rate on the evaluation set.
        """

        num_mini_batches = int(np.ceil(len(game_files) / self.batch_size))
        section_mini_batches = [
            game_files[i*self.batch_size : (i+1)*self.batch_size]
            for i in range(num_mini_batches) if game_files[i*self.batch_size : (i+1)*self.batch_size]
        ]

        if not section_mini_batches:
            logger.warning(f"No games to evaluate for {eval_type}.")
            return

        # Optional: switch shrinkage_lambda_max for eval phase to favor region
        # utility over per-memory Q. Only applies if val_lambda_max is set AND
        # region_manager exists. Restored in finally to keep training lambda intact.
        # Sentinel pattern distinguishes "attribute missing" from "attribute set to None"
        # so restoration can correctly del vs re-set (per Codex review).
        rm = getattr(self.memory_service, 'region_manager', None)
        val_lmax = getattr(self, 'val_lambda_max', None)
        _MISSING = object()
        saved_lmax = _MISSING
        if val_lmax is not None and rm is not None:
            saved_lmax = getattr(rm, 'shrinkage_lambda_max', _MISSING)
            rm.shrinkage_lambda_max = float(val_lmax)
            logger.info("[%s] eval: switching shrinkage_lambda_max %s → %.3f",
                        eval_type,
                        saved_lmax if saved_lmax is not _MISSING else "<missing>",
                        float(val_lmax))

        # Switch to eval temperature if override specified (restored in finally)
        _llm = getattr(self.agent, 'llm_provider', None) or getattr(self.agent, 'llm', None)
        saved_temperature = getattr(_llm, 'temperature', 0) if _llm else 0
        eval_temperature = temperature_override if temperature_override is not None else saved_temperature
        if _llm and eval_temperature != saved_temperature:
            _llm.temperature = eval_temperature

        try:
            # Pre-build failure index once before eval loop to avoid per-batch rebuild
            if getattr(self, '_failure_summary_n_slots', 0) > 0:
                self._ensure_failure_index_built()

            # Pre-warm memory key embeddings so parallel retrieve threads don't
            # serially embed missing keys (the biggest remaining bottleneck per Codex).
            self._prewarm_query_embeddings()

            # --- Sample from each environment ---
            eval_trajectories = []
            for i, mini_batch_games in tqdm(enumerate(section_mini_batches), desc=f"Evaluating on {eval_type}"):
                mini_batch_env = None
                try:
                    mini_batch_env = self.envs_built(mini_batch_games, task_type=eval_type)
                    collected_trajs = self._sample_from_batch(
                        mini_batch_env, _repair_task_type=eval_type
                    )
                    # Keep only the fields metric computation needs (success, steps).
                    # Eval is stateless; retaining full trajectory/retrieved_mems for
                    # every game balloons RAM (OOM on 3553-game train eval at bs=32).
                    save_eval_details = os.environ.get(
                        "MEMRL_ALFWORLD_SAVE_EVAL_TRAJECTORIES", "0"
                    ).strip().lower() not in {"0", "false", "no"}
                    for t in collected_trajs:
                        item = {
                            "success": t.get("success", False),
                            "steps": t.get("steps", 0),
                            "gamefile": t.get("gamefile", ""),
                        }
                        if save_eval_details:
                            item.update({
                                "trajectory": t.get("trajectory", []),
                                "retrieved_queries": t.get("retrieved_queries", []),
                                "retrieved_mems": t.get("retrieved_mems", {}),
                                "deferred_repair": t.get("deferred_repair", False),
                                "deferred_repair_round": t.get("deferred_repair_round"),
                                "transport_invalid": t.get("transport_invalid", False),
                                "aborted": t.get("aborted", False),
                                "timeout": t.get("timeout", False),
                            })
                        eval_trajectories.append(item)
                finally:
                    try:
                        if mini_batch_env is not None:
                            mini_batch_env.close()
                    except Exception:
                        logger.debug("Failed to close eval mini_batch_env", exc_info=True)
        finally:
            # Restore train-phase temperature
            if _llm:
                _llm.temperature = saved_temperature
            # Restore train-phase lambda before returning, even on error.
            if val_lmax is not None and rm is not None:
                if saved_lmax is _MISSING:
                    # Attribute didn't exist before — del to fully restore.
                    try:
                        del rm.shrinkage_lambda_max
                    except AttributeError:
                        pass
                else:
                    rm.shrinkage_lambda_max = saved_lmax

        if not eval_trajectories:
            logger.warning(f"No trajectories were collected during {eval_type} evaluation.")
            self.writer.add_scalar(f"Evaluation/Success_Rate/{eval_type}", 0.0, after_section)
            self.writer.add_scalar(f"Evaluation/Avg_Steps/{eval_type}", 0.0, after_section)
            return 0.0

        # --- Calculate metrics and log the results ---
        successes = sum(1 for traj in eval_trajectories if traj["success"])
        success_rate = successes / len(eval_trajectories) if eval_trajectories else 0.0

        # Calculate average steps
        avg_steps = np.mean([traj["steps"] for traj in eval_trajectories])

        logger.info(f"--- Evaluation Complete on {eval_type} (after training Section {after_section}) ---")
        logger.info(f"Success Rate: {success_rate:.2%} ({successes}/{len(eval_trajectories)})")
        logger.info(f"Average Steps on Success: {avg_steps:.2f}") # Also print to console

        # --- Save per-game results for CSR computation ---
        try:
            import json
            per_game_results = []
            for traj in eval_trajectories:
                record = {
                    "gamefile": traj.get("gamefile", ""),
                    "success": bool(traj.get("success", False)),
                    "steps": int(traj.get("steps", 0)),
                }
                for key in (
                    "trajectory", "retrieved_queries", "retrieved_mems",
                    "deferred_repair", "deferred_repair_round",
                    "transport_invalid", "aborted", "timeout",
                ):
                    if key in traj:
                        record[key] = traj.get(key)
                per_game_results.append(record)

            # Keep this in the experiment's durable checkpoint cache, not a
            # transient logger attribute.  Atomic replace prevents a crash from
            # leaving a partially-written per-game artifact behind.
            out_dir = self.local_cache_dir / "per_game_eval"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_suffix = f"_run{run_idx}" if self.n_eval_runs > 1 else ""
            out_path = out_dir / f"{eval_type}_s{after_section}{run_suffix}.json"
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(per_game_results, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, out_path)
            logger.info("[PER-GAME] Saved %d results to %s", len(per_game_results), out_path)
        except Exception as e:
            logger.warning("[PER-GAME] Failed to save: %s", e, exc_info=True)

        # --- [TENSORBOARD] Log both metrics ---
        self.writer.add_scalar(f"Evaluation/Success_Rate/{eval_type}", success_rate, after_section)
        self.writer.add_scalar(f"Evaluation/Avg_Steps/{eval_type}", avg_steps, after_section)

        # Log this result for the final text report
        self.results_log.append({
            "section": f"eval_s{after_section}",
            "after_section": after_section,
            "mode": eval_type,
            "success": success_rate,
            "steps": avg_steps # Store the new metric
        })

        return success_rate


    def run(self):
        """
        The main experiment execution flow, featuring section and batch loops.
        """
        start_section = self._resume_section_start
        start_batch = self._resume_batch_start
        # MED #8: prefer the persisted flag over inferring from section/batch index.
        # If cum_state.json restored _initial_eval_completed=True we know the initial
        # eval has run, regardless of whether we're starting fresh at (1,1) after a
        # partial artifacts loss. Manual --skip_initial_eval still overrides.
        skip_initial_eval = (
            getattr(self, '_initial_eval_completed', False)
            or start_section > 1
            or start_batch > 1
            or getattr(self, 'skip_initial_eval', False)
        )
        if self.baseline_mode in {"passk", "reflection"}:
            if self.baseline_mode == "passk":
                self._run_passk_baseline()
            else:
                self._run_reflection_baseline()
            self.writer.close()
            return
        # Test-only mode: run evaluation and exit
        if self.mode == 'test':
            logger.info("Running in TEST-ONLY mode (no training)")
            # Optional: evaluate on training set too (for no-mem baseline,
            # we want a train SR datapoint comparable to MemRL train SR).
            if getattr(self, 'eval_train_in_test_mode', False) and self.train_game_files:
                logger.info(f"Evaluating on train_set: {len(self.train_game_files)} games")
                self._evaluate(
                    game_files=self.train_game_files,
                    eval_type="eval_train",
                    after_section=0
                )
            # Evaluate on in-distribution eval set. Optional diagnostic filter
            # selects an explicitly declared subset without changing the dataset.
            valid_eval_games = list(self.valid_game_files)
            filter_raw = os.environ.get("MEMRL_ALFWORLD_ID_GAME_FILTER", "").strip()
            if filter_raw:
                filters = [x.strip() for x in filter_raw.split(",") if x.strip()]
                valid_eval_games = [
                    game for game in valid_eval_games
                    if any(token in str(game) for token in filters)
                ]
                logger.info(
                    "[ID DIAGNOSTIC] filtered ID games: %d/%d using %d patterns",
                    len(valid_eval_games), len(self.valid_game_files), len(filters),
                )
                if not valid_eval_games:
                    raise RuntimeError("ID diagnostic filter matched zero games")
            if valid_eval_games:
                logger.info(f"Evaluating on eval_in_distribution: {len(valid_eval_games)} games")
                self._evaluate(
                    game_files=valid_eval_games,
                    eval_type="eval_in_distribution",
                    after_section=0
                )
            # Skip OOD while selecting a checkpoint on ID validation.
            if self.test_game_files and not getattr(self, 'id_eval_only', False):
                logger.info(f"Evaluating on eval_out_of_distribution: {len(self.test_game_files)} games")
                self._evaluate(
                    game_files=self.test_game_files,
                    eval_type="eval_out_of_distribution",
                    after_section=0
                )
            self._analyze_and_report_results()
            self.writer.close()
            return

        # Training mode: run initial eval then training loop
        logger.info("eval split")
        if not skip_initial_eval:
            if self.holdout_subtask and self.holdout_eval_game_files:
                # Holdout mode: initial eval is on holdout subtask (the only metric we care about)
                logger.info(
                    "[HOLDOUT] Initial eval: zero-shot transfer to %s on %d games (section 0)",
                    self.holdout_subtask, len(self.holdout_eval_game_files),
                )
                self._evaluate(
                    game_files=self.holdout_eval_game_files,
                    eval_type=f"holdout_{self.holdout_subtask.replace('alf/', '')}",
                    after_section=0,
                )
            else:
                self._evaluate(
                        game_files=self.valid_game_files,
                        eval_type="eval_in_distribution",
                        after_section=0
                    )
            # Persist the flag so future resumes don't re-run the initial eval.
            self._initial_eval_completed = True
            try:
                self._persist_cum_state()
            except Exception:
                logger.warning("Failed to persist _initial_eval_completed flag", exc_info=True)

    # --- Loop: Iterate through Sections ---
        # 1. Prepare data splits
        train_sections_data = self.envs_spilt(self.train_game_files, 'train')

        # Region clustering uses a cross-section running counter so split/merge
        # cadence is stable regardless of how trajectories distribute across sections.
        # Resume-safe: load from instance attribute if previously persisted, else start at 0.
        if not hasattr(self, '_global_step'):
            self._global_step = 0

        for section_idx, section_data in enumerate(train_sections_data):
            section_num = section_idx + 1
            if section_num < start_section:
                logger.info("Skipping section %d due to resume.", section_num)
                continue

            logger.info("\n" + "#"*20 + f" STARTING SECTION {section_num}/{self.num_section}" + "#"*20)

            # Region: set epoch for exploration schedule
            if hasattr(self.memory_service, 'set_current_epoch'):
                self.memory_service.set_current_epoch(section_num, num_epochs=self.num_section)

            # Register subtask embeddings for zero-shot transfer (once per process).
            # Use a flag rather than section_num==1 so resumed runs starting at
            # section > 1 still register on first entry. The check is process-local;
            # if subtask_embeddings was loaded from a region_manager snapshot,
            # _register_subtask_embeddings is idempotent (skips existing keys).
            if not getattr(self, '_subtasks_registered', False):
                self._register_subtask_embeddings()
                self._subtasks_registered = True

            # A resumed compact snapshot can have a valid memory store but a
            # mostly-missing query-key embedding cache.  Warm it once before
            # the first training retrieval, not only during evaluation.
            if not getattr(self, "_train_query_keys_prewarmed", False):
                self._prewarm_query_embeddings()
                self._train_query_keys_prewarmed = True

            section_trajectories = []

            # --- Inner Loop: Iterate through mini-batches (environments) ---
            for i, mini_batch_games in tqdm(enumerate(section_data)):
                if section_num == start_section and (i + 1) < start_batch:
                    logger.info("Skipping batch %d/%d due to batch-level resume.", i+1, len(section_data))
                    continue
                logger.info(f"Processing mini-batch {i+1}/{len(section_data)} in section {section_num}...")

                # Collect trajectories
                mini_batch_env = None
                # Pre-initialise so a crash inside _sample_from_batch doesn't
                # leak UnboundLocalError that masks the original exception
                # when section_trajectories.extend(collected_trajs) runs below.
                collected_trajs: List[Dict] = []
                try:
                    mini_batch_env = self.envs_built(mini_batch_games, 'train')
                    collected_trajs = self._sample_from_batch(mini_batch_env)
                except Exception:
                    logger.error(
                        "Mini-batch %d/%d failed during env build or sampling; "
                        "skipping this batch with 0 trajectories.",
                        i + 1, len(section_data), exc_info=True,
                    )
                finally:
                    try:
                        if mini_batch_env is not None:
                            mini_batch_env.close()
                    except Exception:
                        # MED #7: upgrade silent close failure to warning so leaked
                        # subprocesses / FDs are visible. Don't re-raise — close
                        # failures must not stop the training loop.
                        logger.warning("Failed to close mini_batch_env", exc_info=True)

                logger.info(f"Mini-batch {i+1} collected {len(collected_trajs)} trajectories.")

                # Per-batch diagnostic summary
                n_success = sum(1 for t in collected_trajs if t.get('success'))
                n_total = len(collected_trajs)
                avg_steps = np.mean([t.get('steps', 0) for t in collected_trajs]) if collected_trajs else 0
                logger.info(
                    f"[BATCH SUMMARY] mini-batch {i+1}/{len(section_data)}: "
                    f"success={n_success}/{n_total} ({100*n_success/max(1,n_total):.1f}%), "
                    f"avg_steps={avg_steps:.1f}"
                )
                if not collected_trajs:
                    # Sampling failed completely — skip downstream learning paths
                    # for this batch and proceed to the next. cum_state still
                    # persisted at the bottom of the loop so progress is tracked.
                    continue
                section_trajectories.extend(collected_trajs)

                # Aborted trajectories (env crashed mid-episode) carry empty obs
                # and reward=0; including them in memory writes or Q updates would
                # poison the cube with fake failure exemplars and pull region Q
                # toward 0. They are kept in section_trajectories for accounting,
                # but excluded from all learning paths below.
                clean_trajs = [t for t in collected_trajs if not t.get('aborted', False)]
                n_aborted = len(collected_trajs) - len(clean_trajs)
                if n_aborted:
                    logger.warning(
                        f"Mini-batch {i+1}: {n_aborted}/{len(collected_trajs)} trajectories "
                        f"aborted (env error). Excluded from memory writes and Q updates."
                    )

                # --- Memory Processing for this mini-batch ---
                task_descriptions = [traj["task_description"] for traj in clean_trajs]
                trajectories = [traj['trajectory'] for traj in clean_trajs]
                successes = [traj["success"] for traj in clean_trajs]

                retrieved_ids_list = [
                    [
                        mem["memory_id"]
                        for mem_list in traj["retrieved_mems"].values()
                        for mem in mem_list
                        if "memory_id" in mem
                    ]
                    for traj in clean_trajs
                ]

                retrieved_queries = [traj["retrieved_queries"] for traj in clean_trajs]

                # Build target_subtasks from task_type for region Q updates.
                # If region is enabled, target_subtasks=None silently disables
                # region utility updates — fail loudly so we don't ship a
                # "region experiment" that's actually a baseline.
                target_subtasks = None
                rm_for_strict = getattr(self.memory_service, 'region_manager', None)
                try:
                    from memrl.configs.task_hierarchy import get_primary_subtask
                    target_subtasks = [
                        get_primary_subtask("alfworld", {
                            "task_type": traj.get("task_type", ""),
                            "game_file": traj.get("gamefile", ""),
                        })
                        for traj in clean_trajs
                    ]
                except Exception as e:
                    logger.error(
                        "get_primary_subtask import/call failed; "
                        "region experiments will silently degrade to baseline. Error: %s",
                        e, exc_info=True,
                    )
                    if rm_for_strict is not None:
                        # In a region experiment this is a correctness bug —
                        # silent degradation would invalidate the experiment.
                        raise RuntimeError(
                            "Cannot build target_subtasks for region update; aborting."
                        ) from e

                # update q value for retrieved mems
                updated_q_list = self.memory_service.update_values(
                    successes, retrieved_ids_list, target_subtasks=target_subtasks,
                )
                logger.info(f"Updated Q-values for mini-batch {i+1}: {updated_q_list}")

                # Region: mid-epoch clustering maintenance
                # self._global_step is a cross-section running counter (persisted in
                # cum_state.json for resume safety) so the 1500-trajectory cluster-init
                # threshold and 1200-trajectory split/merge cadence apply globally.
                # Only count clean (non-aborted) trajectories so the cadence reflects
                # actual learning samples.
                self._global_step += len(clean_trajs)
                rm = getattr(self.memory_service, 'region_manager', None)
                _CLUSTER_INIT_STEP = getattr(self, '_region_cluster_init_step', 1500)
                _CLUSTER_MERGE_INTERVAL = getattr(self, '_region_merge_interval', 1200)
                _MID_EPOCH_TOPOLOGY = getattr(self, '_region_mid_epoch_topology_enabled', True)
                _COOLDOWN_SECTIONS = getattr(self, '_region_topology_cooldown_sections', 0)
                if rm is not None:
                    if not rm._is_clustered and self._global_step >= _CLUSTER_INIT_STEP:
                        try:
                            rm.cluster_by_utility()
                            self._last_region_topology_section = section_num
                            logger.info(f"[alf] section {section_num} INITIAL cluster at step {self._global_step}: {len(rm.regions)} regions")
                        except Exception as e:
                            logger.warning(f"Initial clustering failed: {e}")
                    elif rm._is_clustered:
                        for mem_id in rm.subtask_q:
                            if mem_id not in rm.membership_weights:
                                rm.assign_new_memory(mem_id)
                        crossed_interval = (
                            self._global_step >= _CLUSTER_INIT_STEP
                            and (self._global_step // _CLUSTER_MERGE_INTERVAL)
                            > ((self._global_step - len(clean_trajs)) // _CLUSTER_MERGE_INTERVAL)
                        )
                        last_edit = getattr(self, '_last_region_topology_section', -1)
                        cooldown_ok = (_COOLDOWN_SECTIONS <= 0 or section_num - last_edit > _COOLDOWN_SECTIONS)
                        if _MID_EPOCH_TOPOLOGY and crossed_interval and cooldown_ok:
                            try:
                                changed = rm.maybe_split_merge()
                                if changed:
                                    self._last_region_topology_section = section_num
                                    logger.info(f"[alf] section {section_num} split/merge at step {self._global_step}: {len(rm.regions)} regions")
                            except Exception as e:
                                logger.warning(f"Split/merge failed: {e}")

                metadatas_update = [
                    {
                        "source_benchmark": "alfworld_build",
                        "success": traj["success"],
                        "task_type": traj.get("task_type", ""),
                        "source_subtask": get_primary_subtask("alfworld", {
                            "task_type": traj.get("task_type", ""),
                            "game_file": traj.get("gamefile", ""),
                        }),
                        "q_value": float(self.rl_config.q_init_pos) if traj['success'] else float(self.rl_config.q_init_neg),
                        "q_visits": 0,
                        "q_updated_at": datetime.now().isoformat(),
                        "last_used_at": datetime.now().isoformat(),
                        "reward_ma": 0.0,
                    }
                    for traj in clean_trajs
                ]

                self.memory_service.add_memories(
                    task_descriptions=task_descriptions,
                    trajectories=trajectories,
                    successes=successes,
                    retrieved_memory_queries=retrieved_queries,
                    retrieved_memory_ids_list=retrieved_ids_list,
                    metadatas=metadatas_update
                )

                logger.info(f"Mini-batch {i+1} memory update complete.")

                # Save checkpoint periodically (every N batches) instead of every batch.
                # Per-batch full snapshots (cube.dump + qdrant copytree + mos.get_all)
                # were the primary OOM amplifier — 927409 hit 4.3TB VMSize at batch 77
                # before SIGKILL. cum_state.json is still written every batch so resume
                # granularity drops only to nearest N batches.
                _BATCH_CKPT_INTERVAL = getattr(self, '_batch_ckpt_interval', 10)
                is_last_batch = (i + 1) == len(section_data)
                should_snapshot = ((i + 1) % _BATCH_CKPT_INTERVAL == 0) or is_last_batch
                try:
                    self._update_cum_success(collected_trajs)
                    self._persist_cum_state()
                    if should_snapshot:
                        batch_ckpt_id = f"s{section_num}_b{i+1}"
                        self.memory_service.save_checkpoint_snapshot(
                            self.ck_dir, ckpt_id=batch_ckpt_id
                        )
                        # Co-locate cum_state.json with snapshot so resume reads a consistent
                        # pair (snapshot at batch K + cum_state at batch K). Without this,
                        # snapshot-at-K + cum_state-at-K+r causes _global_step / success_ids
                        # to skip over replayed batches, leading to double-counted memories
                        # when those batches re-sample new UUIDs into the cube.
                        snapshot_cum_state = Path(self.ck_dir) / "snapshot" / batch_ckpt_id / "local_cache" / "cum_state.json"
                        try:
                            self._persist_cum_state(snapshot_cum_state)
                        except Exception:
                            logger.warning("Failed to co-locate cum_state in snapshot %s", batch_ckpt_id, exc_info=True)
                        logger.info(f"Batch checkpoint saved: {batch_ckpt_id}")
                        # Keep only the latest N batch checkpoints to avoid disk bloat
                        self._cleanup_old_batch_checkpoints(
                            section_num,
                            current_batch=i + 1,
                            max_keep=getattr(self, "_batch_ckpt_keep", 3),
                        )
                        # Force GC after the heavy snapshot path released its big temporaries
                        # (cube serialization, mos.get_all materialization, copytree buffers).
                        import gc
                        gc.collect()
                except Exception:
                    logger.warning(f"Failed to save batch checkpoint at batch {i+1}", exc_info=True)

                if self._stop_after_batch and (i + 1) >= self._stop_after_batch:
                    logger.info(
                        "[SHORT WINDOW] Reached configured stop batch %d after completing batch %d; "
                        "ending this section window.", self._stop_after_batch, i + 1,
                    )
                    break

            logger.info(f"Section {section_num} complete. Total {len(section_trajectories)} trajectories collected.")

            # Region: end-of-section recluster (split/merge)
            # Skip if no new trajectories (eval-only resume) to avoid state drift
            rm = getattr(self.memory_service, 'region_manager', None)
            if rm is not None and len(section_trajectories) > 0:
                try:
                    cluster_init_step = getattr(self, '_region_cluster_init_step', 1500)
                    cooldown_sections = getattr(self, '_region_topology_cooldown_sections', 0)
                    if not rm._is_clustered:
                        if self._global_step >= cluster_init_step:
                            rm.cluster_by_utility()
                            self._last_region_topology_section = section_num
                        else:
                            logger.info(
                                "Section %d: deferred initial region cluster (global_step=%d < init_step=%d)",
                                section_num, self._global_step, cluster_init_step,
                            )
                    else:
                        for mem_id in rm.subtask_q:
                            if mem_id not in rm.membership_weights:
                                rm.assign_new_memory(mem_id)
                        last_edit = getattr(self, '_last_region_topology_section', -1)
                        cooldown_ok = (cooldown_sections <= 0 or section_num - last_edit > cooldown_sections)
                        if cooldown_ok:
                            changed = rm.maybe_split_merge()
                            if changed:
                                self._last_region_topology_section = section_num
                        else:
                            logger.info(
                                "Section %d: skipped region split/merge due to %d-section cooldown after section %d",
                                section_num, cooldown_sections, last_edit,
                            )
                    rm.classify_transfer_patterns()
                    logger.info(
                        "Section %d: region clustering done. %d regions, %d memories.",
                        section_num, len(rm.regions), len(rm.membership_weights),
                    )
                except Exception as e:
                    logger.warning(f"Region clustering failed at section {section_num}: {e}")

            self._update_cum_success(section_trajectories)
            cum_acc = self._current_cum_acc()
            self._persist_cum_state()
            logger.info("Section %d Cumulative Acc: %.2f%%", section_num, cum_acc * 100)
            self.writer.add_scalar("Train/Cumulative_Success_Rate", cum_acc, section_num)
            self.results_log.append({
                "section": section_num,
                "mode": "train_cumulative",
                "success": cum_acc,
                "steps": None,
            })

            # Skip checkpoint save when no new training data (eval-only resume):
            # the region state is unchanged, re-saving would be a no-op that risks
            # overwriting a clean checkpoint with rebuilt artifacts. See docs/RESUME_EVAL_DRIFT.md
            if len(section_trajectories) > 0:
                try:
                    ckpt_meta = self.memory_service.save_checkpoint_snapshot(
                        self.ck_dir, ckpt_id=section_num
                    )
                except Exception as e:
                    logger.warning("Failed to save section checkpoint: %s", e, exc_info=True)
                    ckpt_meta = {}
                logger.info(f" Saved ckpt: {ckpt_meta}")
                snapshot_root = Path(self.ck_dir) / "snapshot" / str(section_num)
                self._persist_cum_state(snapshot_root / "local_cache" / "cum_state.json")
            else:
                logger.info("Eval-only section (0 trajectories): skipping checkpoint save")
            # --- Log results for this section ---
            for traj_data in section_trajectories:
                self.results_log.append({
                    "section": section_num,
                    "success": traj_data["success"],
                    "steps": traj_data["steps"],
                })

            # --- [TENSORBOARD] Log training metrics for this section ---
            if section_trajectories:
                section_success = sum(1 for traj in section_trajectories if traj["success"])
                section_success_rate = section_success / len(section_trajectories)
                section_avg_steps = np.mean([traj["steps"] for traj in section_trajectories if traj['trajectory']])
                
                self.writer.add_scalar("Train/Section_Success_Rate", section_success_rate, section_num)
                self.writer.add_scalar("Train/Section_Avg_Steps", section_avg_steps, section_num)
                logger.info(f"Section {section_num} Training Stats: Success Rate={section_success_rate:.2%}, Avg Steps={section_avg_steps:.2f}")

            if self.mode != 'test':
                # Holdout mode: only run holdout eval, skip the 6-subtask in-distribution
                # valid eval entirely (it's irrelevant to the zero-shot transfer question).
                if self.holdout_subtask and self.holdout_eval_game_files:
                    # Trigger holdout eval at every section (matches valid_interval=1 cadence)
                    # or at valid_interval cadence if user set valid_interval > 1.
                    holdout_cadence = max(1, self.valid_interval) if self.valid_interval > 0 else 1
                    if section_num % holdout_cadence == 0:
                        logger.info(
                            "[HOLDOUT] Evaluating zero-shot transfer to %s on %d games",
                            self.holdout_subtask, len(self.holdout_eval_game_files),
                        )
                        self._evaluate(
                            game_files=self.holdout_eval_game_files,
                            eval_type=f"holdout_{self.holdout_subtask.replace('alf/', '')}",
                            after_section=section_num,
                        )
                else:
                    # Non-holdout mode: regular in-distribution valid eval
                    if self.valid_interval > 0 and section_num % self.valid_interval == 0:
                        self._evaluate(
                            game_files=self.valid_game_files,
                            eval_type="eval_in_distribution",
                            after_section=section_num
                        )

                # Check if it's time to run evaluation on the OOD test set (valid_unseen).
                # This is the standard "OOD" eval for ALFWorld (unseen room layouts).
                if self.test_interval > 0 and section_num % self.test_interval == 0:
                    self._evaluate(
                        game_files=self.test_game_files,
                        eval_type="eval_out_of_distribution",
                        after_section=section_num
                    )

            # Final analysis at the end of all sections
        self._analyze_and_report_results()
        # --- [TENSORBOARD] Close the writer ---
        self.writer.close()
