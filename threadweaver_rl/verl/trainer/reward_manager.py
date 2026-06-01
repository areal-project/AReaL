import torch
import threading
import os
import multiprocessing as mp
import time
import numpy as np

from verl import DataProto
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from deepscaler.rewards.math_rewardv2 import deepscaler_reward_fn

# --- Caching Setup ---
# A simple in-memory cache for reward computation results. This dictionary is
# shared among threads. For multiprocessing, each process gets its own copy,
# providing per-process caching.
_REWARD_CACHE = {}
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 1000000  # 1M
# --- End Caching Setup ---

# print_verbose = print
print_verbose = lambda *args, **kwargs: None


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _cfg_float(config, key: str, default: float = 0.0) -> float:
    try:
        if hasattr(config, "get"):
            return _safe_float(config.get(key, default), default)
    except Exception:
        pass
    try:
        return _safe_float(config[key], default)
    except Exception:
        return default


def _groupwise_zscore(values: np.ndarray, group_ids: list[str], eps: float) -> np.ndarray:
    z = np.zeros_like(values, dtype=np.float32)
    groups: dict[str, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(idx)

    for idxs in groups.values():
        group_vals = values[idxs]
        mu = float(np.mean(group_vals))
        sigma = float(np.std(group_vals))
        if sigma <= eps:
            z[idxs] = 0.0
        else:
            z[idxs] = (group_vals - mu) / sigma
    return z


def _apply_groupwise_parallel_bonus(
    scores: list[dict],
    extra_infos: list[dict],
    uid_values,
    config,
) -> tuple[list[dict], list[dict]]:
    beta_1 = _cfg_float(config, "subtask_beta", 0.0)
    beta_2 = _cfg_float(config, "trial_beta", 0.0)
    beta_3 = _cfg_float(config, "parallel_ratio_beta", 0.0)
    alpha = _cfg_float(config, "latency_alpha", 0.0)
    eps = max(1e-12, _cfg_float(config, "group_shaping_eps", 1e-8))

    n = len(scores)
    if len(extra_infos) != n:
        extra_infos = list(extra_infos)[:n] + [{} for _ in range(max(0, n - len(extra_infos)))]

    if uid_values is None:
        group_ids = [str(i) for i in range(n)]
    else:
        try:
            uid_list = uid_values.tolist() if hasattr(uid_values, "tolist") else list(uid_values)
        except Exception:
            uid_list = [uid_values] * n
        if len(uid_list) != n:
            group_ids = [str(i) for i in range(n)]
        else:
            group_ids = [str(uid) for uid in uid_list]

    subtask_ratio = np.array([_safe_float((extra_infos[i] or {}).get("subtask_ratio", 0.0)) for i in range(n)], dtype=np.float32)
    trial_ratio = np.array([_safe_float((extra_infos[i] or {}).get("trial_ratio", 0.0)) for i in range(n)], dtype=np.float32)
    parallel_ratio = np.array([_safe_float((extra_infos[i] or {}).get("parallel_ratio", 0.0)) for i in range(n)], dtype=np.float32)
    latency_signal = np.array([_safe_float((extra_infos[i] or {}).get("acceleration_ratio", 0.0)) for i in range(n)], dtype=np.float32)

    subtask_z = _groupwise_zscore(subtask_ratio, group_ids, eps=eps)
    trial_z = _groupwise_zscore(trial_ratio, group_ids, eps=eps)
    parallel_z = _groupwise_zscore(parallel_ratio, group_ids, eps=eps)
    latency_z = _groupwise_zscore(latency_signal, group_ids, eps=eps)

    parallel_bonus = beta_1 * subtask_z + beta_2 * trial_z + beta_3 * parallel_z + alpha * latency_z

    bonus_scores: list[dict] = []
    bonus_extra_infos: list[dict] = []
    for i in range(n):
        score = scores[i] if isinstance(scores[i], dict) else {"reward": _safe_float(scores[i]), "second_reward": 0.0}
        score = dict(score)
        base_reward = _safe_float(score.get("reward", 0.0))
        info = dict(extra_infos[i] or {})
        bonus_if_correct = float(parallel_bonus[i]) if bool(info.get("correct", False)) else 0.0
        score["reward"] = base_reward + bonus_if_correct
        bonus_scores.append(score)

        info["parallel_bonus_beta_1"] = beta_1
        info["parallel_bonus_beta_2"] = beta_2
        info["parallel_bonus_beta_3"] = beta_3
        info["parallel_bonus_alpha"] = alpha
        info["parallel_bonus_subtask_z"] = float(subtask_z[i])
        info["parallel_bonus_trial_z"] = float(trial_z[i])
        info["parallel_bonus_parallel_ratio_z"] = float(parallel_z[i])
        info["parallel_bonus_latency_z"] = float(latency_z[i])
        info["parallel_rewardv2_bonus_raw"] = float(parallel_bonus[i])
        info["parallel_rewardv2_bonus"] = bonus_if_correct
        info["reward_before_parallel_bonus"] = base_reward
        info["reward_after_parallel_bonus"] = score["reward"]
        bonus_extra_infos.append(info)

    return bonus_scores, bonus_extra_infos


def _select_rm_score_fn(data_source):
    return deepscaler_reward_fn

def process_item(args):
    global _REWARD_CACHE

    i, data_item, tokenizer, config, num_examine, correctness_as_reward, skip_reward_fn = args

    print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] Processing {i}")
    prompt_ids = data_item.batch["prompts"]
    prompt_length = prompt_ids.shape[-1]

    valid_prompt_length = data_item.batch["attention_mask"][
        :prompt_length
    ].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch["responses"]
    valid_response_length = data_item.batch["attention_mask"][
        prompt_length:
    ].sum()
    valid_response_ids = response_ids[:valid_response_length]

    # decode
    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    sequences_str = tokenizer.decode(sequences)

    print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] sequences for {i}")


    # with open(f"/mnt/wsfuse/longlian/POLARIS/sequence_dumps/{os.environ['EXPERIMENT_NAME']}_{os.environ.get('GLOBAL_STEP', '?')}_{i}.txt", "w") as f:
    #     f.write(sequences_str)

    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    data_source = data_item.non_tensor_batch["data_source"]

    # --- Caching Logic ---
    # Create a hashable key. Using str(ground_truth) ensures hashability even if it's a list or dict.
    # We only cache results from successful (non-skipped) computations.
    cache_key = (sequences_str, str(ground_truth), data_source, correctness_as_reward)

    # If not in a fallback/timeout scenario, check the cache first.
    if not skip_reward_fn:
        with _CACHE_LOCK:
            cached_result = _REWARD_CACHE.get(cache_key)
        # If a result is found in the cache, return it immediately.
        if cached_result:
            score, extra_info = cached_result
            if i < num_examine:
                print("[sequence] (cached)", sequences_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)
            return i, score, valid_response_length, sequences_str, extra_info
    # --- End Caching Logic ---

    # If it's a cache miss or we're skipping the reward function, compute the score.
    compute_score_fn = _select_rm_score_fn(data_source)
    # Filter out client-only keys that RewardConfig does not accept
    cfg = config or {}
    try:
        cfg_keys = list(cfg.keys()) if hasattr(cfg, "keys") else list(cfg)
    except Exception:
        cfg_keys = []
    rm_cfg = {}
    for k in cfg_keys:
        ks = str(k)
        if ks.startswith("length_penalty_"):
            continue
        rm_cfg[ks] = cfg[k]

    score, extra_info = compute_score_fn(
        solution_str=sequences_str,
        ground_truth=ground_truth,
        config=rm_cfg,
        correctness_as_reward=correctness_as_reward,
        skip_reward_fn=skip_reward_fn,
        tokenizer=tokenizer,
    )

    # --- Caching Logic ---
    # Store the result in the cache only if it was a full, successful computation.
    if not skip_reward_fn:
        with _CACHE_LOCK:
            if len(_REWARD_CACHE) >= _MAX_CACHE_SIZE:
                _REWARD_CACHE = {}
            _REWARD_CACHE[cache_key] = (score, extra_info)
    # --- End Caching Logic ---

    if i < num_examine:
        # print("[prompt]", prompt_str)
        # print("[response]", response_str)
        print("[sequence]", sequences_str)
        print("[ground_truth]", ground_truth)
        print("[score]", score)

    # Uncomment this to test the timeout:
    # if not skip_reward_fn:
    #     import time
    #     time.sleep(100000)

    print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] Finished processing {i}")

    return i, score, valid_response_length, sequences_str, extra_info

def process_batch(batch_args):
    batch_out = []
    for arg in batch_args:
        batch_out.append(process_item(arg))
    return batch_out

class RewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config = config

    def __call__(self, data: DataProto, return_dict: bool = False, correctness_as_reward: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If rm scores are already attached, reuse them while optionally returning extra info.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                meta_info = data.meta_info or {}
                reward_extra_keys = meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch.get(key)
                    for key in reward_extra_keys
                    if key in data.non_tensor_batch
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        secondary_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        if True:
            # Use threads (cannot cancel threads once they're run)
            # Note that the threads that are really stuck will not be cancelled, and will block program exit, but it should not interfere with training.
            executor = ThreadPoolExecutor(max_workers=96)
            args = [
                (i, data[i], self.tokenizer, self.config, self.num_examine, correctness_as_reward, False)
                for i in range(len(data))
            ]
            future_to_arg = {executor.submit(process_item, arg): arg for arg in args}

            results = [None] * len(args)
            for fut, arg in future_to_arg.items():
                i = arg[0]
                try:
                    print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] Waiting for result for", i)
                    res = fut.result(timeout=3)
                    print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] Result obtained for", i)
                except FuturesTimeout:
                    fut.cancel()  # best-effort
                    print(f"Timeout for {i}, skip reward fn")
                    arg_skip = (i, arg[1], self.tokenizer, self.config, self.num_examine, correctness_as_reward, True)
                    res = process_item(arg_skip)
                    print(f"Timeout for {i}, skip reward fn, using result (skipping reward function): {res}")
                except Exception as e:
                    arg_skip = (i, arg[1], self.tokenizer, self.config, self.num_examine, correctness_as_reward, True)
                    res = process_item(arg_skip)
                    print(f"Error for {i}, skip reward fn, response: {res[3]}, error: {e}")
                results[i] = res

            print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] All results obtained from the reward function.")

            executor.shutdown(wait=False, cancel_futures=True)

        if False:
            # Use multiprocessing (can cancel processes once they're run, but may have overhead)
            args = [
                (i, data[i], self.tokenizer, self.config, self.num_examine, correctness_as_reward, False)
                for i in range(len(data))
            ]

            # We'll collect results in correct order
            results = [None] * len(args)

            # Use fork context (Linux), hf tokenizer parallelism will be disabled
            ctx = mp.get_context("fork")

            # This avoids the hf tokenizer parallelism, which is not compatible with multiprocessing + fork.
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            with ctx.Pool(processes=48) as pool:
                asyncs = [pool.apply_async(process_item, (a,)) for a in args]

                for idx, (fut, a) in enumerate(zip(asyncs, args)):
                    i = a[0]
                    try:
                        res = fut.get(timeout=3)  # seconds
                    except mp.context.TimeoutError:
                        # Kill the worker process handling this task by terminating pool and respawning a new one is heavy;
                        # simpler: re-run locally with skip_reward_fn=True (fast path), and mark timeout.
                        # Optionally log here; printing in subprocess is flaky.
                        print(f"Timeout for {i}, skip reward fn.", flush=True)

                        # Fallback compute without heavy reward fn
                        arg_skip = (i, a[1], self.tokenizer, self.config, self.num_examine, correctness_as_reward, True)
                        res = process_item(arg_skip)
                        print(f"Timeout for {i}, skip reward fn, using result (skipping reward function): {res}", flush=True)
                    except Exception as e:
                        print(f"Worker error on {i}: {e}, skip reward fn.", flush=True)
                        arg_skip = (i, a[1], self.tokenizer, self.config, self.num_examine, correctness_as_reward, True)
                        res = process_item(arg_skip)
                        print(f"Worker error on {i}, skip reward fn, using result (skipping reward function): {res}", flush=True)

                    results[i] = res

                pool.terminate()

            del os.environ["TOKENIZERS_PARALLELISM"]

        if False:
            # Use multiprocessing (can cancel processes once they're run, with chunks)
            # NOTE: once something times out (which should be rare), the chunk will be skipped (skip_reward_fn=True).
            # 1) Build per‐example args
            args = [
                (i, data[i], self.tokenizer, self.config, self.num_examine, correctness_as_reward, False)
                for i in range(len(data))
            ]
            # 2) Split into small batches to amortize IPC overhead
            chunk_size = 32
            batches = [args[i : i + chunk_size] for i in range(0, len(args), chunk_size)]
            # placeholder for all results
            results = [None] * len(args)

            # 3) Prepare multiprocessing context
            ctx = mp.get_context("fork")
            os.environ["TOKENIZERS_PARALLELISM"] = "true"

            # 4) Launch & collect with timeout + fallback
            with ctx.Pool(processes=48) as pool:
                asyncs = [pool.apply_async(process_batch, (batch,)) for batch in batches]

                for batch, async_res in zip(batches, asyncs):
                    try:
                        batch_results = async_res.get(timeout=3)
                    except mp.context.TimeoutError:
                        # fallback: run each example locally skipping heavy reward fn
                        batch_results = []
                        for i, data_item, tok, cfg, num_ex, corr, _ in batch:
                            fallback_arg = (i, data_item, tok, cfg, num_ex, corr, True)
                            batch_results.append(process_item(fallback_arg))

                    # write back into the global results list
                    for i, score, valid_response_length, sequences_str, extra_info in batch_results:
                        results[i] = (i, score, valid_response_length, sequences_str, extra_info)

                pool.terminate()

            # 6) Cleanup
            del os.environ["TOKENIZERS_PARALLELISM"]

        # print("Done with all results")

        scores = [res[1] for res in results]
        extra_infos = [res[4] or {} for res in results]
        uid_values = data.non_tensor_batch.get("uid")
        scores, extra_infos = _apply_groupwise_parallel_bonus(scores, extra_infos, uid_values, self.config)

        # Fill reward tensor with results
        for idx, (i, _, valid_response_length, _, _) in enumerate(results):
            score = scores[idx] if isinstance(scores[idx], dict) else {"reward": _safe_float(scores[idx]), "second_reward": 0.0}
            reward_tensor[i, valid_response_length - 1] = _safe_float(score.get("reward", 0.0))
            secondary_reward_tensor[i, valid_response_length - 1] = _safe_float(score.get("second_reward", 0.0))

        # Turn extra_infos from list of dicts to dict of lists
        extra_infos_dict = {}
        for extra_info in extra_infos:
            for key, value in extra_info.items():
                if key not in extra_infos_dict:
                    extra_infos_dict[key] = []
                extra_infos_dict[key].append(value)

        # Return reward tensor and extra info dictionary
        if return_dict:
            return {
                "reward_tensor": {
                    "main_reward_tensor": reward_tensor,
                    "secondary_reward_tensor": secondary_reward_tensor,
                },
                "reward_extra_info": extra_infos_dict,
            }
        else:
            return reward_tensor
