# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Cross-Benchmark Memory Transfer Experiment - Python Version

A flexible Python implementation that supports:
1. Training memory on any benchmark
2. Transferring trained memory to other benchmarks for testing
3. Automatic analysis and comparison of results

Usage:
    python run_cross_benchmark_experiment.py \
        --source llb \
        --targets hle bcb \
        --api_key YOUR_API_KEY \
        --mode srun

Can also be imported as a module.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import logging

# Add project path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    source_benchmark: str
    target_benchmarks: List[str]
    experiment_name: str = ""

    # API configuration (默认使用 LiteLLM 本地服务)
    llm_api_key: str = ""
    llm_base_url: str = "http://127.0.0.1:4000"
    llm_model: str = "gpt-4o-2024-11-20"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"

    # Training configuration
    num_sections: int = 10
    batch_size: int = 5
    random_seed: int = 42

    # LLB specific configuration
    llb_task: str = "os"  # os or db

    # SLURM configuration
    partition: str = "all"
    gpus: int = 1
    cpus: int = 8
    memory: str = "32G"
    time_limit: str = "24:00:00"

    # Output configuration
    results_dir: str = ""
    config_dir: str = ""

    def __post_init__(self):
        if not self.experiment_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"cross_{self.source_benchmark}_{timestamp}"

        if not self.results_dir:
            self.results_dir = str(PROJECT_ROOT / "results")

        if not self.config_dir:
            self.config_dir = str(PROJECT_ROOT / "scripts" / "generated_configs")

        if not self.embedding_api_key:
            self.embedding_api_key = self.llm_api_key

        if not self.embedding_base_url:
            self.embedding_base_url = self.llm_base_url


class CrossBenchmarkExperiment:
    """Cross-Benchmark Memory Transfer Experiment Manager"""

    # Similarity normalization params for different benchmarks
    SIM_NORM_PARAMS = {
        "llb_os": {"mean": 0.39, "std": 0.14},
        "llb_db": {"mean": 0.27, "std": 0.11},
        "bcb": {"mean": 0.31, "std": 0.10},
        "hle": {"mean": 0.19, "std": 0.09},
        "alf": {"mean": 0.52, "std": 0.12},
        "webshop": {"mean": 0.35, "std": 0.12},
    }

    # Benchmark runner scripts
    BENCHMARK_SCRIPTS = {
        "llb": "run_llb.py",
        "bcb": "run_bcb.py",
        "hle": "run_hle.py",
        "alf": "run_alfworld.py",
        "webshop": "run_webshop.py",
    }

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.checkpoint_path: Optional[str] = None
        self.results: Dict[str, Any] = {}

        # Ensure directories exist
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.config_dir).mkdir(parents=True, exist_ok=True)

    def _get_split_file(self, benchmark: str, mode: str) -> str:
        """Get the appropriate data file path for each benchmark"""
        if benchmark == "llb":
            return f"data/llb/{self.config.llb_task}_interaction_train.json"
        elif benchmark == "hle":
            return "data/hle/hle_test.parquet"
        elif benchmark == "alf":
            return ""  # ALFWorld uses config-based data loading
        elif benchmark == "bcb":
            return ""  # BCB uses its own data loading mechanism
        elif benchmark == "webshop":
            return ""  # WebShop uses its own data loading
        return ""

    def generate_config(
        self,
        benchmark: str,
        mode: str,  # "train" or "eval"
        checkpoint_path: Optional[str] = None
    ) -> str:
        """Generate benchmark configuration file"""
        config_path = Path(self.config.config_dir) / f"{self.config.experiment_name}_{benchmark}_{mode}.yaml"

        # Get similarity normalization params
        if benchmark == "llb":
            sim_key = f"llb_{self.config.llb_task}"
        else:
            sim_key = benchmark
        sim_params = self.SIM_NORM_PARAMS.get(sim_key, {"mean": 0.3, "std": 0.1})

        config_dict = {
            "llm": {
                "provider": "openai",
                "api_key": self.config.llm_api_key,
                "base_url": self.config.llm_base_url,
                "model": self.config.llm_model,
                "temperature": 0.0,
                "max_tokens": 10240,
            },
            "embedding": {
                "provider": "openai",
                "api_key": self.config.embedding_api_key,
                "base_url": self.config.embedding_base_url,
                "model": self.config.embedding_model,
                "max_text_len": 6000,  # Keep under 8192 tokens (~4 chars/token)
            },
            "memory": {
                "build_strategy": "proceduralization",
                "retrieve_strategy": "query",
                "update_strategy": "adjustment",
                "k_retrieve": 10,
                "max_keywords": 8,
                "confidence_threshold": 0.0,
                "memory_confidence": 100.0,
                "add_similarity_threshold": 0.99,
                "mos_config_path": "configs/mos_config.json",
                "user_id": f"{self.config.experiment_name}_{benchmark}",
                "load_from_checkpoint": bool(checkpoint_path),
                "checkpoint_path": checkpoint_path,
                "sim_norm_mean": sim_params["mean"],
                "sim_norm_std": sim_params["std"],
            },
            "environment": {
                "alfworld_config_path": "configs/envs/alfworld.yaml",
                "alfworld_env_type": "AlfredTWEnv",
            },
            "experiment": {
                "experiment_name": f"{self.config.experiment_name}_{benchmark}_{mode}",
                "algorithm": "rl",
                "val_before_train": False,
                "enable_value_driven": True,
                "random_seed": self.config.random_seed,
                "mode": "train" if mode == "train" else "test",
                "task": self.config.llb_task if benchmark == "llb" else benchmark,
                "split_file": self._get_split_file(benchmark, mode),
                "valid_file": None,
                "num_sections": self.config.num_sections if mode == "train" else 1,
                "batch_size": self.config.batch_size,
                "max_steps": 15,
                "valid_interval": 0,
                "test_interval": 1,
                "dataset_ratio": 1.0,
                "few_shot_path": "data/alfworld/alfworld_examples.json",
                "bon": 0,
                "hle_categories": None,
                "hle_category_ratio": None,
                "ckpt_eval_enabled": False,
                "ckpt_eval_path": None,
                "ckpt_resume_enabled": False,
                "ckpt_resume_path": None,
                "ckpt_resume_epoch": None,
                "baseline_mode": None,
                "baseline_k": 10,
                "output_dir": self.config.results_dir,
                "save_trajectories": True,
                "save_memories": True,
                "enable_logging": True,
                "log_level": "INFO",
            },
            "rl_config": {
                "epsilon": 0.01 if mode == "train" else 0.0,
                "tau": 0.35,
                "alpha": 0.3,
                "gamma": 0.0,
                "q_init_pos": 0.5,
                "q_init_neg": 0.5,
                "success_reward": 1.0,
                "failure_reward": 0.0,
                "sim_threshold": 0.5,
                "topk": 5,
                "novelty_threshold": 0.85,
                "recency_boost": 0.0,
                "reward_merge_gain": 0.1,
                "q_min_threshold": -0.8,
                "weight_sim": 0.5,
                "weight_q": 0.5,
            },
        }

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Config file generated: {config_path}")
        return str(config_path)

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint"""
        results_path = Path(self.config.results_dir)
        source = self.config.source_benchmark

        # Find matching experiment directories
        patterns = [
            results_path / source / f"exp_{self.config.experiment_name}*",
            results_path / f"*{source}*" / f"exp_{self.config.experiment_name}*",
        ]

        for pattern in patterns:
            exp_dirs = sorted(pattern.parent.glob(pattern.name), key=lambda x: x.stat().st_mtime, reverse=True)
            for exp_dir in exp_dirs:
                snapshot_dir = exp_dir / "snapshot"
                if snapshot_dir.exists():
                    # Find latest epoch
                    epochs = sorted(snapshot_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                    if epochs:
                        checkpoint_path = str(epochs[0])
                        logger.info(f"Found checkpoint: {checkpoint_path}")
                        return checkpoint_path

        logger.warning("No checkpoint found")
        return None

    def run_benchmark(
        self,
        benchmark: str,
        config_path: str,
        use_srun: bool = False
    ) -> bool:
        """Run a single benchmark"""
        script = self.BENCHMARK_SCRIPTS.get(benchmark)
        if not script:
            logger.error(f"Unknown benchmark: {benchmark}")
            return False

        cmd = ["python3", "-B", str(PROJECT_ROOT / "run" / script), "--config", config_path]

        # BCB needs extra parameters
        if benchmark == "bcb":
            cmd.extend(["--subset", "hard", "--epochs", str(self.config.num_sections)])

        if use_srun:
            srun_cmd = [
                "srun",
                f"--partition={self.config.partition}",
                f"--gres=gpu:{self.config.gpus}",
                f"--cpus-per-task={self.config.cpus}",
                f"--mem={self.config.memory}",
                f"--time={self.config.time_limit}",
            ] + cmd
            cmd = srun_cmd

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=False,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return False

    def run_training(self, use_srun: bool = False) -> bool:
        """Run training phase"""
        logger.info(f"===== Training phase: {self.config.source_benchmark} =====")

        config_path = self.generate_config(self.config.source_benchmark, "train")
        success = self.run_benchmark(self.config.source_benchmark, config_path, use_srun)

        if success:
            # Wait for filesystem sync
            time.sleep(5)
            self.checkpoint_path = self.find_latest_checkpoint()

        return success and self.checkpoint_path is not None

    def run_evaluation(self, target: str, use_srun: bool = False) -> bool:
        """Run evaluation phase"""
        logger.info(f"===== Evaluation phase: {target} =====")

        if not self.checkpoint_path:
            logger.error("No checkpoint available")
            return False

        config_path = self.generate_config(target, "eval", self.checkpoint_path)
        return self.run_benchmark(target, config_path, use_srun)

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results"""
        try:
            from scripts.analyze_cross_benchmark_results import generate_report, print_report, save_report
        except ImportError:
            logger.warning("Could not import analysis module")
            return {}

        report = generate_report(
            results_dir=self.config.results_dir,
            experiment_name=self.config.experiment_name,
            source_benchmark=self.config.source_benchmark,
            target_benchmarks=self.config.target_benchmarks,
        )

        print_report(report)

        # Save report
        report_path = Path(self.config.results_dir) / f"cross_benchmark_report_{self.config.experiment_name}.json"
        save_report(report, str(report_path))

        return report

    def run(self, use_srun: bool = False) -> bool:
        """Run complete experiment"""
        logger.info("=" * 60)
        logger.info("Cross-Benchmark Memory Transfer Experiment")
        logger.info("=" * 60)
        logger.info(f"Source Benchmark: {self.config.source_benchmark}")
        logger.info(f"Target Benchmarks: {self.config.target_benchmarks}")
        logger.info(f"Experiment name: {self.config.experiment_name}")
        logger.info("=" * 60)

        # Step 1: Training
        if not self.run_training(use_srun):
            logger.error("Training phase failed")
            return False

        # Step 2: Evaluate each target benchmark
        for target in self.config.target_benchmarks:
            if not self.run_evaluation(target, use_srun):
                logger.warning(f"Evaluation of {target} failed, continuing with others...")

        # Step 3: Analyze results
        try:
            self.analyze_results()
        except Exception as e:
            logger.warning(f"Result analysis failed: {e}")

        logger.info("=" * 60)
        logger.info("Experiment completed!")
        logger.info("=" * 60)

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Benchmark Memory Transfer Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on LLB, test on HLE and BCB
  python run_cross_benchmark_experiment.py --source llb --targets hle bcb --api_key YOUR_KEY

  # Use srun to run on compute nodes
  python run_cross_benchmark_experiment.py --source llb --targets hle --api_key YOUR_KEY --mode srun

  # Specify model
  python run_cross_benchmark_experiment.py --source llb --targets hle --api_key YOUR_KEY --model gpt-4o
        """
    )

    parser.add_argument("--source", required=True, choices=["llb", "bcb", "hle", "alf", "webshop"],
                        help="Source benchmark (train memory)")
    parser.add_argument("--targets", required=True, nargs="+", choices=["llb", "bcb", "hle", "alf", "webshop"],
                        help="Target benchmarks (test memory transfer)")
    parser.add_argument("--api_key", required=True, help="LLM API key")
    parser.add_argument("--base_url", default="http://127.0.0.1:4000", help="API base URL (LiteLLM)")
    parser.add_argument("--model", default="gpt-4o-2024-11-20", help="Model name")
    parser.add_argument("--embedding_model", default="text-embedding-3-small", help="Embedding model")
    parser.add_argument("--name", default="", help="Experiment name (auto-generated by default)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--llb_task", default="os", choices=["os", "db"], help="LLB task type")
    parser.add_argument("--mode", default="local", choices=["local", "srun"],
                        help="Run mode: local or srun (compute node)")

    # SLURM configuration
    parser.add_argument("--partition", default="all", help="SLURM partition")
    parser.add_argument("--gpus", type=int, default=1, help="GPU count")
    parser.add_argument("--cpus", type=int, default=8, help="CPU count")
    parser.add_argument("--mem", default="32G", help="Memory")
    parser.add_argument("--time", default="24:00:00", help="Time limit")

    args = parser.parse_args()

    config = ExperimentConfig(
        source_benchmark=args.source,
        target_benchmarks=args.targets,
        experiment_name=args.name,
        llm_api_key=args.api_key,
        llm_base_url=args.base_url,
        llm_model=args.model,
        embedding_model=args.embedding_model,
        num_sections=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
        llb_task=args.llb_task,
        partition=args.partition,
        gpus=args.gpus,
        cpus=args.cpus,
        memory=args.mem,
        time_limit=args.time,
    )

    experiment = CrossBenchmarkExperiment(config)
    success = experiment.run(use_srun=(args.mode == "srun"))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
