# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
import json
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.trainer.generation_io import (
    prepare_output_paths,
    check_existing_jsonl_complete,
    resume_from_jsonl,
    open_jsonl_append,
    write_jsonl_record,
    write_json_file,
    completion_stats,
)
from verl.trainer.reward_stats import (
    compute_reward_stats,
    format_comprehensive_stats_table,
    format_eval_results_table,
)
from verl.experimental.agent_loop import AgentLoopManager


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def _select(cfg, key, default=None):
    """Lightweight OmegaConf.select wrapper with a safe default.

    Returns the selected node if present; otherwise returns `default`.
    """
    try:
        val = OmegaConf.select(cfg, key)
        return val if val is not None else default
    except Exception:
        return default


def _ensure_min_actor_cfg(node):
    """Ensure minimal actor config fields exist for rollout workers.

    ActorRolloutRefWorker unconditionally reads a few actor fields (e.g., fsdp_size,
    ulysses_sequence_parallel_size). This helper injects safe defaults when absent.
    """
    if node is None:
        node = OmegaConf.create({})
    with open_dict(node):
        if "strategy" not in node or node.strategy is None:
            node["strategy"] = "fsdp"
        if "ulysses_sequence_parallel_size" not in node or node.ulysses_sequence_parallel_size is None:
            node["ulysses_sequence_parallel_size"] = 1
        if "ppo_mini_batch_size" not in node:
            node["ppo_mini_batch_size"] = None
        if "use_dynamic_bsz" not in node:
            node["use_dynamic_bsz"] = False
        if "ppo_max_token_len_per_gpu" not in node:
            node["ppo_max_token_len_per_gpu"] = 16384
        fsdp_cfg = node.get("fsdp_config", None)
        if fsdp_cfg is None:
            node["fsdp_config"] = {"fsdp_size": -1, "forward_prefetch": False, "use_orig_params": False}
        else:
            if "fsdp_size" not in fsdp_cfg or fsdp_cfg.fsdp_size is None:
                node.fsdp_config["fsdp_size"] = -1
            if "forward_prefetch" not in fsdp_cfg:
                node.fsdp_config["forward_prefetch"] = False
            if "use_orig_params" not in fsdp_cfg:
                node.fsdp_config["use_orig_params"] = False
    return node


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    # Try to pretty print resolved config; fallback to non-resolved if resolution fails
    try:
        pprint(OmegaConf.to_container(config, resolve=True))
    except Exception:
        pprint(OmegaConf.to_container(config, resolve=False))

    # Resolve new-style vs old-style locations for model/rollout configs
    rollout_cfg = _select(config, "actor_rollout_ref.rollout", default=_select(config, "rollout"))
    model_cfg = _select(config, "actor_rollout_ref.model", default=_select(config, "model"))
    if model_cfg is None or rollout_cfg is None:
        raise ValueError(
            "Missing model/rollout config. Expected actor_rollout_ref.model/rollout (new) or model/rollout (legacy)."
        )

    # Validate sampling constraints
    if rollout_cfg.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # Read dataset early for resume/skip checks
    dataset = pd.read_parquet(config.data.path)
    # Optional dataset splitting to match reference script
    total_splits = int(_select(config, "data.total_splits", 1) or 1)
    current_split = int(_select(config, "data.current_split", 0) or 0)
    if current_split >= total_splits:
        raise ValueError(
            f"current_split ({current_split}) must be less than total_splits ({total_splits})"
        )
    if current_split < 0:
        raise ValueError(f"current_split ({current_split}) must be non-negative")
    if total_splits < 1:
        raise ValueError(f"total_splits ({total_splits}) must be at least 1")

    if total_splits > 1:
        total_rows = len(dataset)
        split_size = (total_rows + total_splits - 1) // total_splits
        start_idx = current_split * split_size
        end_idx = min(start_idx + split_size, total_rows)
        dataset = dataset.iloc[start_idx:end_idx].reset_index(drop=True)
        print(
            f"Dataset splitting: Processing split {current_split}/{total_splits-1}"
        )
        print(
            f"Original dataset size: {total_rows}, current split size: {len(dataset)} (rows {start_idx}-{end_idx-1})"
        )

    total_samples = len(dataset)
    n_samples = int(config.data.n_samples)

    # If targeting JSONL outputs, check if an existing JSONL is complete and skip model init entirely.
    output_path = os.path.expanduser(str(config.data.output_path))
    use_jsonl_outputs = not output_path.lower().endswith(".parquet")

    local_path = copy_to_local(model_cfg.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if use_jsonl_outputs:
        results_dir, jsonl_path, json_path, _ = prepare_output_paths(
            model_path=model_cfg.path,
            output_root=output_path,
            dataset=dataset,
            dataset_path=config.data.path,
            n_samples=n_samples,
            total_splits=total_splits,
            current_split=current_split,
        )
        if check_existing_jsonl_complete(
            jsonl_path=jsonl_path,
            json_path=json_path,
            results_dir=results_dir,
            total_messages=total_samples,
            n_samples=n_samples,
        ):
            # If JSONL is already complete, optionally compute reward stats before exiting.
            try:
                if "reward_model" in dataset.columns:
                    with open(json_path, "r", encoding="utf-8") as _fh:
                        responses = json.load(_fh)

                    # Load tokenizer to enable v2 reward metrics that need token IDs
                    local_path = copy_to_local(model_cfg.path)
                    trust_remote_code = config.data.get("trust_remote_code", False)
                    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

                    stats = compute_reward_stats(dataset=dataset, responses=responses, tokenizer=tokenizer)
                    print("\n--- Evaluation Results ---")
                    print(format_eval_results_table(stats))
                    print("\n--- Comprehensive Statistics ---")
                    print(format_comprehensive_stats_table(stats))
            except Exception as e:
                print(f"Reward stats computation skipped due to error: {e}")
                import traceback
                traceback.print_exc()
            return

    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_async_rollout = str(rollout_cfg.mode).lower() == "async"

    # Build worker config in PPO-style under actor_rollout_ref; inject minimal actor defaults
    actor_rollout_ref_cfg = _select(config, "actor_rollout_ref")
    if actor_rollout_ref_cfg is None:
        actor_rollout_ref_cfg = OmegaConf.create({})
    with open_dict(actor_rollout_ref_cfg):
        if "model" not in actor_rollout_ref_cfg or actor_rollout_ref_cfg.model is None:
            actor_rollout_ref_cfg["model"] = model_cfg
        if "rollout" not in actor_rollout_ref_cfg or actor_rollout_ref_cfg.rollout is None:
            actor_rollout_ref_cfg["rollout"] = rollout_cfg
        # Ensure minimal actor config
        actor_cfg = actor_rollout_ref_cfg.get("actor")
        actor_rollout_ref_cfg["actor"] = _ensure_min_actor_cfg(actor_cfg)

    worker_cls = AsyncActorRolloutRefWorker if use_async_rollout else ActorRolloutRefWorker
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(worker_cls), config=actor_rollout_ref_cfg, role="rollout"
    )
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    agent_loop_manager = None
    pad_divisor = wg.world_size
    if use_async_rollout:
        # Use the complete config like PPO trainer does, with safe defaults for missing pieces
        agent_cfg = OmegaConf.create(config)
        OmegaConf.set_struct(agent_cfg, False)

        # Ensure required top-level fields exist
        if not hasattr(agent_cfg, "reward_model") or agent_cfg.reward_model is None:
            agent_cfg.reward_model = OmegaConf.create({"enable": False, "enable_resource_pool": False})
        if not hasattr(agent_cfg, "global_profiler") or agent_cfg.global_profiler is None:
            agent_cfg.global_profiler = OmegaConf.create({"tool": None})

        # Ensure actor_rollout_ref subtree exists and is self-contained
        if not hasattr(agent_cfg, "actor_rollout_ref") or agent_cfg.actor_rollout_ref is None:
            agent_cfg.actor_rollout_ref = OmegaConf.create({})
        with open_dict(agent_cfg.actor_rollout_ref):
            if "model" not in agent_cfg.actor_rollout_ref or agent_cfg.actor_rollout_ref.model is None:
                agent_cfg.actor_rollout_ref["model"] = model_cfg
            if "rollout" not in agent_cfg.actor_rollout_ref or agent_cfg.actor_rollout_ref.rollout is None:
                agent_cfg.actor_rollout_ref["rollout"] = rollout_cfg
            # Ensure minimal actor config present
            agent_cfg.actor_rollout_ref["actor"] = _ensure_min_actor_cfg(
                agent_cfg.actor_rollout_ref.get("actor")
            )

            # Ensure rollout.agent exists
            if (
                not hasattr(agent_cfg.actor_rollout_ref.rollout, "agent")
                or agent_cfg.actor_rollout_ref.rollout.agent is None
            ):
                agent_cfg.actor_rollout_ref.rollout.agent = OmegaConf.create(
                    {
                        "num_workers": 1,
                        "agent_loop_config_path": None,
                        "enable_parallel_branching": False,
                        "no_conclusion": False,
                        "verbose": 0,
                    }
                )

        agent_loop_manager = AgentLoopManager(config=agent_cfg, worker_group=wg, rm_wg=None)
        pad_divisor = max(1, len(agent_loop_manager.agent_loop_workers))

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    num_batch = -(-total_samples // config_batch_size)
    # For JSONL rich outputs (tokens + generated_blocks)
    total_messages = total_samples
    n_samples = int(config.data.n_samples)

    # Progressive JSONL writing + resume support (non-parquet path)
    if use_jsonl_outputs:
        # Prepare resume map from JSONL if present
        responses_by_message, _ = resume_from_jsonl(
            jsonl_path=jsonl_path, total_messages=total_messages, n_samples=n_samples
        )
    else:
        # Placeholder when writing Parquet only; still build responses at the end from in-memory results
        responses_by_message = [[None] * n_samples for _ in range(total_messages)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        start = batch_idx * config_batch_size
        end = (batch_idx + 1) * config_batch_size
        batch_chat_lst = chat_lst[start:end]

        # Build DataProto for this batch
        if use_async_rollout:
            # Build non-tensor inputs for agent loop. Besides the raw prompt, also forward
            # optional reward/eval-related fields if they exist in the dataset so that
            # reward computation (when enabled) has the required metadata.
            non_tensor_batch = {
                "raw_prompt": np.array(batch_chat_lst, dtype=object),
            }

            # Pass through commonly used optional columns if present
            optional_keys = ["data_source", "reward_model", "extra_info", "uid"]
            for key in optional_keys:
                if key in dataset.columns:
                    # Keep object dtype to preserve nested structures (dict/list)
                    non_tensor_batch[key] = np.array(dataset[key].iloc[start:end].tolist(), dtype=object)

            meta_info = {"validate": False, "global_steps": -1, "return_generated_blocks": True}
            data_full = DataProto(batch=None, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
        else:
            inputs = tokenizer.apply_chat_template(
                batch_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=int(rollout_cfg.prompt_length),
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

            data_full = DataProto.from_dict(batch_dict)

        # Flatten (prompt, sample) pairs: generate all pending pairs for this batch in one go
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        global_indices = list(range(start, min(end, total_samples)))

        # Build task list of (local_i, sample_idx, global_idx) for all missing samples
        tasks = []
        for local_i, global_idx in enumerate(global_indices):
            for sample_idx in range(n_samples):
                if responses_by_message[global_idx][sample_idx] is None:
                    tasks.append((local_i, sample_idx, global_idx))

        if not tasks:
            continue

        # Select subset with repeats for multiple samples from the same prompt
        repeat_local_indices = [t[0] for t in tasks]
        data_subset = data_full.select_idxs(repeat_local_indices)
        gen_batch_size = len(data_subset)
        data_padded, pad_size = pad_dataproto_to_divisor(data_subset, pad_divisor)
        data_padded.meta_info["return_expanded_sequences"] = False
        data_padded.meta_info["compute_length_penalty"] = False

        if use_async_rollout:
            output_padded = agent_loop_manager.generate_sequences(data_padded)
        else:
            output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        assert len(output) == gen_batch_size, f"{len(output)=} != {gen_batch_size=}"

        # Append results for these tasks to JSONL and in-memory cache
        if use_jsonl_outputs:
            jsonl_fh = open_jsonl_append(jsonl_path)
        else:
            jsonl_fh = None
        try:
            for out_i in range(len(output)):
                data_item = output[out_i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=False)

                local_i, sample_idx, global_idx = tasks[out_i]

                # Update in-memory cache
                responses_by_message[global_idx][sample_idx] = response_str

                # Progressive JSONL write
                if jsonl_fh is not None:
                    generated_blocks = None
                    if use_async_rollout:
                        ntb = getattr(data_item, "non_tensor_batch", {}) or {}
                        generated_blocks = ntb.get("generated_blocks", None)
                    record = {
                        "message_idx": global_idx,
                        "sample_idx": sample_idx,
                        "result": response_str,
                        "generated_blocks": generated_blocks,
                    }
                    write_jsonl_record(jsonl_fh, record)
        finally:
            if jsonl_fh is not None:
                jsonl_fh.close()

    # Build dataset responses from in-memory cache
    dataset["responses"] = [["" if v is None else str(v) for v in row] for row in responses_by_message]

    # Save outputs
    # If output_path ends with .parquet, keep existing parquet behavior.
    # Otherwise, treat it as an output root and write JSONL + JSON bundles
    # similar to scripts/convert_generation_parquet.py.
    if not use_jsonl_outputs:
        output_dir = os.path.dirname(output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(output_path)

        print(f"Generation finished. Results saved to {output_path}")
        print(dataset.head())
        print(dataset['responses'][0][0])
        # Optional: compute reward stats when reward_model.ground_truth is present
        try:
            if "reward_model" in dataset.columns:
                stats = compute_reward_stats(
                    dataset=dataset, responses=dataset["responses"].tolist(), tokenizer=tokenizer
                )
                print("\n--- Evaluation Results ---")
                print(format_eval_results_table(stats))
                print("\n--- Comprehensive Statistics ---")
                print(format_comprehensive_stats_table(stats))
        except Exception as e:
            print(f"Reward stats computation skipped due to error: {e}")
            import traceback
            traceback.print_exc()
        return

    # JSONL has been appended progressively; optionally write JSON when complete
    # Compute completion and write JSON aggregate only when there are no missing entries
    missing, total_entries = completion_stats(responses_by_message)

    if missing == 0:
        write_json_file(json_path, dataset["responses"].tolist())
        print(
            f"Generation finished. Wrote {total_entries} samples for {len(responses_by_message)} prompts (JSON complete)"
        )
        # Compute reward stats when complete and reward_model.ground_truth available
        try:
            if "reward_model" in dataset.columns:
                stats = compute_reward_stats(
                    dataset=dataset, responses=dataset["responses"].tolist(), tokenizer=tokenizer
                )
                print("\n--- Evaluation Results ---")
                print(format_eval_results_table(stats))
                print("\n--- Comprehensive Statistics ---")
                print(format_comprehensive_stats_table(stats))
        except Exception as e:
            print(f"Reward stats computation skipped due to error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(
            f"Generation incomplete: {missing} missing of {total_entries + missing}. JSONL updated; JSON not written."
        )
    print(f"JSONL: {jsonl_path}")
    if missing == 0:
        print(f"JSON:  {json_path}")
    print(dataset.head())
    try:
        print(dataset['responses'][0][0])
    except Exception:
        pass


if __name__ == "__main__":
    main()
