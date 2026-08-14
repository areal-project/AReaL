#!/usr/bin/env python3
"""
生成评估配置文件的辅助脚本
"""

import argparse
import yaml
import os
from pathlib import Path


def generate_eval_config(
    benchmark: str,
    experiment_name: str,
    checkpoint_path: str,
    output_path: str,
    llm_api_key: str = None,
    llm_base_url: str = None,
    llm_model: str = None,
    embedding_api_key: str = None,
    embedding_base_url: str = None,
    embedding_model: str = None,
):
    """生成评估配置文件"""

    # 从环境变量获取默认值
    llm_api_key = llm_api_key or os.environ.get("LLM_API_KEY", "your-api-key")
    llm_base_url = llm_base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model = llm_model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    embedding_api_key = embedding_api_key or os.environ.get("EMBEDDING_API_KEY", llm_api_key)
    embedding_base_url = embedding_base_url or os.environ.get("EMBEDDING_BASE_URL", llm_base_url)
    embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

    # 不同benchmark的归一化参数
    sim_norm_params = {
        "llb_os": {"mean": 0.39, "std": 0.14},
        "llb_db": {"mean": 0.27, "std": 0.11},
        "bcb": {"mean": 0.31, "std": 0.10},
        "hle": {"mean": 0.19, "std": 0.09},
        "alf": {"mean": 0.52, "std": 0.12},
    }

    sim_params = sim_norm_params.get(benchmark, {"mean": 0.3, "std": 0.1})

    config = {
        "llm": {
            "provider": "openai",
            "api_key": llm_api_key,
            "base_url": llm_base_url,
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": 10240,
        },
        "embedding": {
            "provider": "openai",
            "api_key": embedding_api_key,
            "base_url": embedding_base_url,
            "model": embedding_model,
            "max_text_len": 8196,
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
            "user_id": f"{experiment_name}_{benchmark}",
            "load_from_checkpoint": True,
            "checkpoint_path": checkpoint_path,
            "sim_norm_mean": sim_params["mean"],
            "sim_norm_std": sim_params["std"],
        },
        "environment": {
            "alfworld_config_path": "configs/envs/alfworld.yaml",
            "alfworld_env_type": "AlfredTWEnv",
        },
        "experiment": {
            "experiment_name": f"{experiment_name}_{benchmark}_eval",
            "algorithm": "rl",
            "val_before_train": False,
            "enable_value_driven": True,
            "random_seed": 42,
            "mode": "test",
            "task": "os",
            "split_file": "data/llb/os_interaction_data.json",
            "valid_file": None,
            "num_sections": 1,  # 评估只运行1轮
            "batch_size": 5,
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
            "output_dir": "./results",
            "save_trajectories": True,
            "save_memories": True,
            "enable_logging": True,
            "log_level": "INFO",
        },
        "rl_config": {
            "epsilon": 0.0,  # 评估时关闭探索
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

    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"[INFO] 配置文件已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成评估配置文件")
    parser.add_argument("--benchmark", required=True, help="目标benchmark")
    parser.add_argument("--experiment_name", required=True, help="实验名称")
    parser.add_argument("--checkpoint_path", required=True, help="Checkpoint路径")
    parser.add_argument("--output", required=True, help="输出配置文件路径")
    parser.add_argument("--llm_api_key", default=None)
    parser.add_argument("--llm_base_url", default=None)
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--embedding_api_key", default=None)
    parser.add_argument("--embedding_base_url", default=None)
    parser.add_argument("--embedding_model", default=None)

    args = parser.parse_args()

    generate_eval_config(
        benchmark=args.benchmark,
        experiment_name=args.experiment_name,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
