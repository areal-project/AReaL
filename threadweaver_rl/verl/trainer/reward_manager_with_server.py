"""
Adapted Reward Manager that uses a local HTTP reward server.

- Spawns the reward server via `gunicorn` in a subprocess.
- Talks to the server at /compute_reward with JSON payloads containing token IDs.
- Keeps the original RewardManagerWithServer call signature and behavior as much as possible.
- Maintains client-side caching (in addition to server-side caching).
- Cleans up (terminates) the server on object deletion.

Expected server start command (what this module will run internally):

    TOKENIZER_PATH=<your_tokenizer_path> PYTHONPATH=verl gunicorn --workers 4 --bind localhost:8159 "verl.trainer.reward_server:create_app()" --timeout 60

__main__:
- Mirrors the example client: reads a .jsonl file with fields like "output", "ground_truth", "correct".
- Uses a Hugging Face tokenizer's `encode` to turn strings into token ids for a minimal FakeData batch.
"""

import os
import sys
import json
import time
import atexit
import random
import signal
import socket
import threading
import subprocess
from typing import Any, Dict, Tuple, Optional, List

import requests
from requests import HTTPError
from requests.adapters import HTTPAdapter
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from verl import DataProto  # type: ignore

# Example: PYTHONPATH=verl python verl/verl/trainer/reward_manager_with_server.py ~/storage_nao/POLARIS/checkpoints/deepscaler/16n-p1-40k-pt175825-4bd-pfrv2.3npfe_a0.1af10_noegr_fif2-0902_004052/val/650.jsonl -n 128
# PYTHONPATH=verl python verl/verl/trainer/reward_manager_with_server.py ../POLARIS/checkpoints/deepscaler/16n-p1-40k-pt175825-4bd-pfrv2.3npfe_a0.1af10_noegr_fif2-0902_004052/val/650.jsonl -n 128

# print_verbose = print
print_verbose = lambda *args, **kwargs: None


def _safe_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _cfg_float(config: Dict[str, Any], key: str, default: float = 0.0) -> float:
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
    config: Dict[str, Any],
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
    latency_signal = np.array([_safe_float((extra_infos[i] or {}).get("total_num_tokens", 0.0)) for i in range(n)], dtype=np.float32)

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


# =========================
# Client-side Caching Setup
# =========================

_REWARD_CACHE: Dict[Tuple[Tuple[int, ...], str, str, bool, Tuple[Tuple[str, Any], ...]], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 1_000_000  # 1M


# =========================
# Reward Server Management
# =========================

# Server configs
_DEFAULT_BIND_HOST = "localhost"
## This is the timeout for server worker
_DEFAULT_TIMEOUT = 60
_APP_IMPORT_PATH = "verl.trainer.reward_server:create_app()"

def _is_port_open(host: str, port: int) -> bool:
    """Return True if TCP socket is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        try:
            sock.connect((host, port))
            return True
        except Exception:
            return False


def _find_unused_port(host: str = "localhost", start_port: int = 10000, end_port: int = 60000, max_attempts: int = 1000) -> int:
    """Find an unused port starting from start_port."""
    for _ in range(max_attempts):
        port = random.randint(start_port, end_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find an unused port in range {start_port}-{end_port}")


def _wait_for_http_ready(base_url: str, timeout_s: int = 300) -> None:
    """
    Wait until the HTTP server at base_url responds to any request.
    We accept any status code response (e.g., 200/404) as "ready".
    """
    start = time.time()
    sess = requests.Session()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            # Try a GET to root; even a 404 indicates the server is up.
            r = sess.get(base_url, timeout=10.0)
            if r.status_code >= 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"Reward server didn't become ready at {base_url} within {timeout_s}s. Last error: {last_err}")


def _start_reward_server(
    tokenizer_path: str,
    bind_host: str,
    port: int,
    workers: int,
    timeout_s: int,
) -> subprocess.Popen:
    """
    Launch the reward server via gunicorn as a subprocess.
    Returns the Popen object.
    """
    env = os.environ.copy()
    env["TOKENIZER_PATH"] = tokenizer_path
    # Disable tokenizer parallelism in server as requested in server code
    env["TOKENIZERS_PARALLELISM"] = "false"

    bind_arg = f"{bind_host}:{port}"
    cmd = [
        "gunicorn",
        "--workers",
        str(workers),
        "--bind",
        bind_arg,
        _APP_IMPORT_PATH,
        "--timeout",
        str(timeout_s),
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,
        stderr=None,
        text=True,
        start_new_session=True,  # so we can terminate the whole process group
    )

    # Wait until the TCP port is open before moving to HTTP readiness probe
    start = time.time()
    while time.time() - start < timeout_s:
        if _is_port_open(bind_host, port):
            break
        time.sleep(1.0)
    else:
        # If port never opened, dump stderr for debugging
        try:
            _, err_out = proc.communicate(timeout=0.5)
        except Exception:
            err_out = ""
        raise RuntimeError(f"Gunicorn failed to open port {bind_arg}. Stderr:\n{err_out}")

    # HTTP readiness check (accepts 200 or 404, just making sure WSGI stack is responding)
    base_url = f"http://{bind_host}:{port}/"
    _wait_for_http_ready(base_url, timeout_s=timeout_s)

    print(f"Reward server started at {base_url}")
    return proc


def _terminate_process_tree(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    """Terminate gunicorn process group gracefully, then kill if still alive."""
    if proc.poll() is not None:
        return
    try:
        # Send SIGTERM to the whole process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        # fallback: terminate just the proc
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1.0)

    # Force kill
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# =========================
# HTTP Client to Server
# =========================

def _call_reward_server(
    session: requests.Session,
    url: str,
    sequence_ids: List[int],
    ground_truth: Any,
    data_source: str,
    correctness_as_reward: bool,
    skip_reward_fn: bool,
    effective_config: Dict[str, Any],
    timeout_s: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Call the /compute_reward endpoint on the reward server.

    Returns:
        (score_dict, extra_info_dict)
    Raises:
        requests.exceptions.RequestException on network/HTTP errors/timeouts
        ValueError on unexpected payloads
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "sequence_ids": sequence_ids,
        "ground_truth": ground_truth,
        "data_source": data_source,
        "correctness_as_reward": correctness_as_reward,
        "skip_reward_fn": skip_reward_fn,
        "config": effective_config,
    }

    print_verbose(f"Server request: {payload}")
    resp = session.post(url, data=json.dumps(payload), headers=headers, timeout=timeout_s)

    try:
        resp.raise_for_status()
    except HTTPError as e:
        resp_obj = e.response  # may be None for some network errors, but rarely with HTTPError
        if resp_obj is not None:
            try:
                err_json = resp_obj.json()
                print("[HTTPError] Server returned JSON:", err_json)
            except ValueError:
                # Not JSON; show text (truncate to keep logs sane)
                print("[HTTPError] Server returned non-JSON body (truncated):", (resp_obj.text or "")[:2000])
        else:
            print(f"[HTTPError] No response object available: {e}")

        raise  # re-raise so callers can handle/fallback as they already do

    data = resp.json()

    print_verbose(f"Server response: {data}")

    if not isinstance(data, dict) or "score" not in data or "extra_info" not in data:
        raise ValueError(f"Bad response payload from reward server: {data}")
    return data["score"], data["extra_info"]


# =========================
# Per-item processing (thread workers)
# =========================

def _hashable_config(cfg: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Convert dict config to a stable, hashable tuple of sorted (k, v) pairs."""
    return tuple(sorted(cfg.items()))


def process_item(args):
    """
    Thread worker for a single item.
    This version talks to the HTTP reward server instead of computing locally.
    It handles its own exceptions, including timeouts, and attempts a fallback.
    It is designed to not raise exceptions to the caller.
    """
    global _REWARD_CACHE

    (
        i,
        data_item,
        tokenizer,
        config,
        num_examine,
        correctness_as_reward,
        skip_reward_fn,
        http_session,
        item_timeout_s,
        endpoint_url,
    ) = args

    global_step = os.environ.get("GLOBAL_STEP", "?")
    now = time.time()
    print_verbose(f"[global step: {global_step}, {now}] Processing {i}")

    # ==========
    # Extract ids & masks
    # ==========
    prompt_ids = data_item.batch["prompts"]  # 1D tensor
    prompt_length = prompt_ids.shape[-1]

    attention_mask = data_item.batch["attention_mask"]
    # Sum of mask in the prompt segment (ensures valid prompt length)
    valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch["responses"]
    valid_response_length = int(attention_mask[prompt_length:].sum().item())
    valid_response_ids = response_ids[:valid_response_length]

    # Concatenate prompt + response IDs and convert to a list for JSON payload
    sequences_tensor = torch.cat((valid_prompt_ids, valid_response_ids))
    sequence_ids_list = sequences_tensor.tolist()

    sequences_str = None

    if False:
        if sequences_str is None:
            sequences_str = tokenizer.decode(sequences_tensor)

        print_verbose(f"[global step: {global_step}, {time.time()}] sequences for {i}: {sequences_str}")

        try:
            exp_name = os.environ.get("EXPERIMENT_NAME", "EXPERIMENT")
            dump_dir = os.environ.get("SEQUENCE_DUMP_DIR", "/mnt/wsfuse/longlian/POLARIS/sequence_dumps")
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"{exp_name}_{global_step}_{i}.txt")
            with open(dump_path, "w") as f:
                f.write(sequences_str)
        except Exception:
            pass

    ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    data_source = data_item.non_tensor_batch["data_source"]

    # ==========
    # Client-side Caching (use tuple of IDs for key)
    # ==========
    hashable_cfg = _hashable_config(config or {})
    cache_key = (tuple(sequence_ids_list), str(ground_truth), data_source, correctness_as_reward, hashable_cfg)

    if not skip_reward_fn:
        if sequences_str is None:
            sequences_str = tokenizer.decode(sequences_tensor) # Decode for logging/debugging
        with _CACHE_LOCK:
            cached_result = _REWARD_CACHE.get(cache_key)
            if cached_result:
                score, extra_info = cached_result
                if i < num_examine:
                    print("[sequence] (cached)", sequences_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)
                return i, score, valid_response_length, sequences_str, extra_info

    # ==========
    # Remote compute (HTTP server)
    # ==========
    assert endpoint_url is not None, "Reward server URL is not set."

    # sanitize config keys before sending to server: RewardConfig(**config) on server side
    # does not accept client-only keys like length_penalty_*.
    cfg = config or {}
    try:
        cfg_keys = list(cfg.keys()) if hasattr(cfg, "keys") else list(cfg)
    except Exception:
        cfg_keys = []
    server_cfg = {}
    for k in cfg_keys:
        ks = str(k)
        if ks.startswith("length_penalty_"):
            continue
        server_cfg[ks] = cfg[k]

    try:
        score, extra_info = _call_reward_server(
            http_session,
            url=endpoint_url,
            sequence_ids=sequence_ids_list,
            ground_truth=ground_truth,
            data_source=data_source,
            correctness_as_reward=correctness_as_reward,
            skip_reward_fn=skip_reward_fn,
            effective_config=server_cfg,
            timeout_s=item_timeout_s,
        )
    except Exception as e:
        # If network/server error occurs on the "full" path, fallback to skip path (quick path).
        # This catches requests.exceptions.Timeout, ConnectionError, HTTPError, etc.
        # if not skip_reward_fn:
        #     try:
        #         print(f"[WARN] Item {i}: full reward failed ({e}); using skip_reward_fn=True fallback.")
        #         import traceback
        #         traceback.print_exc()

        #         score, extra_info = _call_reward_server(
        #             http_session,
        #             url=endpoint_url,
        #             sequence_ids=sequence_ids_list,
        #             ground_truth=ground_truth,
        #             data_source=data_source,
        #             correctness_as_reward=correctness_as_reward,
        #             skip_reward_fn=True,
        #             effective_config=config or {},
        #             timeout_s=1.0,  # Use a shorter timeout for the quick fallback
        #         )
        #     except Exception as e2:
        #         # As a last resort, synthesize a neutral score to avoid crashing the trainer.
        #         print(f"[ERROR] Item {i}: both full and skip server calls failed ({e2}). Returning zero rewards.")
        #         import traceback
        #         traceback.print_exc()

        #         score = {"reward": 0.0, "second_reward": 0.0}
        #         extra_info = {"error": "server_unreachable", "details": str(e2)}
        # else:
        #     # We were *already* on skip path and it still failed: synthesize neutral score.
        #     print(f"[ERROR] Item {i}: skip_reward_fn server call failed ({e}). Returning zero rewards.")
        #     score = {"reward": 0.0, "second_reward": 0.0}
        #     extra_info = {"error": "server_unreachable_on_skip", "details": str(e)}

        print(f"[ERROR] Item {i}: server call failed ({e}).")

        import traceback
        traceback.print_exc()

        raise e

    # ==========
    # Update cache on successful compute (only when not skip path)
    # ==========
    if not skip_reward_fn:
        with _CACHE_LOCK:
            if len(_REWARD_CACHE) >= _MAX_CACHE_SIZE:
                _REWARD_CACHE.clear()
            _REWARD_CACHE[cache_key] = (score, extra_info)

    if i < num_examine:
        if sequences_str is None:
            sequences_str = tokenizer.decode(sequences_tensor)
        print("[sequence]", sequences_str)
        print("[ground_truth]", ground_truth)
        print("[score]", score)

    print_verbose(f"[global step: {global_step}, {time.time()}] Finished processing {i}")

    return i, score, valid_response_length, sequences_str, extra_info


def process_batch(batch_args):
    batch_out = []
    for arg in batch_args:
        batch_out.append(process_item(arg))
    return batch_out


# =========================
# Reward Manager
# =========================

class RewardManagerWithServer:
    """The reward manager that talks to the HTTP reward server."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        config: Dict[str, Any],
        *,
        tokenizer_path: Optional[str] = None,
        server_host: str = _DEFAULT_BIND_HOST,
        server_port: Optional[int] = None,
        workers: int = 64,
        client_workers: int = 64,
        server_timeout_s: int = _DEFAULT_TIMEOUT,
        auto_start_server: bool = True,
        item_timeout_s: float = 3.0,
    ) -> None:
        """
        Args:
            tokenizer: HF tokenizer-like object with .decode() and .name_or_path
            num_examine: number of decoded responses to print
            config: dict of reward config to send to the server
            tokenizer_path: explicit path for server's TOKENIZER_PATH. If None, uses tokenizer.name_or_path.
            server_host: host to bind/reach the server
            server_port: port to bind/reach the server
            workers: gunicorn workers
            server_timeout_s: gunicorn worker timeout
            auto_start_server: start the server automatically in a subprocess
            item_timeout_s: timeout in seconds for each individual HTTP reward request
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # config might be DictConfig from OmegaConf
        self.config = dict(config) if config else {}
        self.item_timeout_s = item_timeout_s
        self.client_workers = client_workers

        self.server_host = server_host
        # If no port specified, find an unused one
        if server_port is None:
            self.server_port = _find_unused_port(server_host)
            print(f"[RewardManagerWithServer] No port specified, using unused port: {self.server_port}")
        else:
            self.server_port = server_port
        self._server_proc: Optional[subprocess.Popen] = None

        # Reuse HTTP session across threads (requests is thread-safe for distinct calls)
        self._http_session = requests.Session()
        adapter = HTTPAdapter(pool_maxsize=self.client_workers)
        self._http_session.mount("http://", adapter)
        self._http_session.mount("https://", adapter)

        # Determine tokenizer path for the server.
        # Priority: explicit tokenizer_path > tokenizer.name_or_path > env var > default.
        if tokenizer_path:
            self._tokenizer_path = tokenizer_path
        elif hasattr(tokenizer, 'name_or_path') and tokenizer.name_or_path:
            self._tokenizer_path = tokenizer.name_or_path
        else:
            self._tokenizer_path = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3-8B")

        # Start server if requested
        base_url = f"http://{self.server_host}:{self.server_port}"
        self._endpoint_url = f"{base_url}/compute_reward"

        if auto_start_server:
            print(f"[RewardManagerWithServer] Starting server with tokenizer: {self._tokenizer_path}")
            self._server_proc = _start_reward_server(
                tokenizer_path=self._tokenizer_path,
                bind_host=self.server_host,
                port=self.server_port,
                workers=workers,
                timeout_s=server_timeout_s,
            )

        # Ensure cleanup at process exit (best-effort)
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Best-effort cleanup for the server process."""
        if self._server_proc is not None:
            try:
                _terminate_process_tree(self._server_proc)
            except Exception:
                pass
            self._server_proc = None

    def __del__(self):
        # Ensure server is terminated upon GC
        self._cleanup()

    def __call__(self, data: DataProto, return_dict: bool = False, correctness_as_reward: bool = False):
        """
        Same interface as the original RewardManagerWithServer.
        - Builds reward tensors.
        - Computes rewards (via HTTP server) in parallel with thread pool.
        - Timeout/exception handling is managed within the worker `process_item`.
        """
        # If rm scores already exist, reuse them (and attach extras when requested).
        if "rm_scores" in data.batch.keys():
            uid_values = data.non_tensor_batch.get("uid")
            has_bonus_fields = "reward_before_parallel_bonus" in data.non_tensor_batch

            if has_bonus_fields and uid_values is not None:
                n = len(data)
                ntb = data.non_tensor_batch
                ntb_keys = [k for k, v in ntb.items() if hasattr(v, "__len__") and len(v) == n]
                extra_infos = []
                scores = []
                for i in range(n):
                    info = {k: ntb[k][i] for k in ntb_keys}
                    extra_infos.append(info)
                    scores.append({
                        "reward": _safe_float(info.get("reward_before_parallel_bonus", 0.0)),
                        "second_reward": 0.0,
                    })

                scores, extra_infos = _apply_groupwise_parallel_bonus(
                    scores, extra_infos, uid_values, self.config
                )

                rm_scores = data.batch["rm_scores"].clone()
                if "response_mask" in data.batch.keys():
                    response_mask = data.batch["response_mask"]
                else:
                    responses = data.batch["responses"]
                    attention_mask = data.batch["attention_mask"]
                    response_length = responses.size(1)
                    response_mask = attention_mask[:, -response_length:]
                response_lengths = response_mask.sum(-1).long()

                for i in range(n):
                    pos = response_lengths[i].item() - 1
                    if pos >= 0:
                        rm_scores[i, pos] = _safe_float(scores[i].get("reward", 0.0))

                extra_infos_dict: Dict[str, list] = {}
                for info in extra_infos:
                    for key, value in info.items():
                        extra_infos_dict.setdefault(key, []).append(value)

                if return_dict:
                    return {
                        "reward_tensor": rm_scores,
                        "reward_extra_info": extra_infos_dict,
                    }
                return rm_scores

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

        # Threads
        executor = ThreadPoolExecutor(max_workers=self.client_workers)
        args = [
            (
                i,
                data[i],
                self.tokenizer,
                self.config,
                self.num_examine,
                correctness_as_reward,
                False,
                self._http_session,
                self.item_timeout_s,
                self._endpoint_url,
            )
            for i in range(len(data))
        ]

        future_to_arg = {executor.submit(process_item, arg): arg for arg in args}
        results = [None] * len(args)

        for fut, arg in future_to_arg.items():
            i = arg[0]
            print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] Waiting for result for {i}")
            try:
                # Timeout is now handled by the requests call inside process_item.
                # We wait for the future to complete without a separate timeout here.
                res = fut.result(timeout=self.item_timeout_s + 1.0)
                print_verbose(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] Result obtained for {i}")
            except Exception as e:
                # This is a fallback for unexpected errors inside the worker thread itself.
                # process_item is designed to handle its own exceptions and not raise,
                # so this is a last-resort failsafe.
                print(f"Caught unexpected worker error for {i}, falling back to skip_reward_fn. Error: {e}")
                arg_skip = (
                    i,
                    arg[1],
                    self.tokenizer,
                    self.config,
                    self.num_examine,
                    correctness_as_reward,
                    True,
                    self._http_session,
                    10.0,
                    self._endpoint_url,
                )
                res = process_item(arg_skip)
                print(f"Error for {i}, using skip_reward_fn fallback result: {res}")
                print(f"Error args:\n{args}\nError args end")

            results[i] = res

        print(f"[global step: {os.environ.get('GLOBAL_STEP', '?')}, {time.time()}] All results obtained from the reward server.")
        executor.shutdown(wait=False, cancel_futures=True)

        scores = [res[1] for res in results]
        extra_infos = [res[4] or {} for res in results]
        uid_values = data.non_tensor_batch.get("uid")
        scores, extra_infos = _apply_groupwise_parallel_bonus(scores, extra_infos, uid_values, self.config)

        # Fill reward tensor with results
        for idx, (i, _, valid_response_length, _, _) in enumerate(results):
            score = scores[idx] if isinstance(scores[idx], dict) else {"reward": _safe_float(scores[idx]), "second_reward": 0.0}
            reward_tensor[i, valid_response_length - 1] = _safe_float(score.get("reward", 0.0))
            secondary_reward_tensor[i, valid_response_length - 1] = _safe_float(score.get("second_reward", 0.0))

        # Convert list of dicts into dict of lists
        extra_infos_dict: Dict[str, list] = {}
        for info in extra_infos:
            for k, v in info.items():
                if k not in extra_infos_dict:
                    extra_infos_dict[k] = []
                extra_infos_dict[k].append(v)

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


# =========================
# __main__: JSONL-driven smoke test using tokenizer.encode
# =========================

if __name__ == "__main__":
    """
    Minimal test that:
      1) Starts the reward server (using TOKENIZER_PATH env or default 'Qwen/Qwen3-8B').
      2) Loads a .jsonl file of samples (fields: "output", "ground_truth", "correct") like the example client.
      3) Uses a real HF tokenizer's `encode` to make token IDs.
      4) Builds a REAL DataProto with padded tensors and object non_tensors.
      5) Runs RewardManagerWithServer on the batch and prints a compact dict result.

    Example:
      python adapted_reward_manager.py /path/to/samples.jsonl \
        --num-samples 8 --host localhost --port 8159 \
        --tokenizer-path Qwen/Qwen3-8B --correctness-as-reward 0
    """
    import argparse
    import numpy as np
    from transformers import AutoTokenizer  # Requires `pip install transformers`

    parser = argparse.ArgumentParser(
        description="JSONL-driven RewardManagerWithServer smoke test (uses tokenizer.encode) with REAL DataProto."
    )
    parser.add_argument("file_path", help="Path to the .jsonl file containing test samples.")
    parser.add_argument("--host", type=str, default="localhost", help="Reward server host (default: localhost).")
    parser.add_argument("--port", "-p", type=int, default=8159, help="Reward server port (default: 8159).")
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=1,
        help="Number of samples to test from the file (default: 1). Use 0 to process all samples.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3-8B"),
        help="Tokenizer path/name for both local encode/decode and the server TOKENIZER_PATH.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Gunicorn workers for the reward server (default: 4).")
    parser.add_argument("--server-timeout", type=int, default=60, help="Gunicorn worker timeout (default: 60).")
    parser.add_argument(
        "--correctness-as-reward",
        type=int,
        default=0,
        help="If nonzero, pass correctness_as_reward=True to the server for all items.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="math_test",
        help="data_source string to send with each request (default: math_test).",
    )
    parser.add_argument(
        "--num-examine",
        type=int,
        default=1,
        help="Number of decoded sequences to print (default: 1).",
    )
    args = parser.parse_args()

    # ---- Load samples from JSONL, mirroring the example client ----
    # Expect per line JSON with at least:
    #   {
    #     "output": "<model solution text>",
    #     "ground_truth": <anything JSON-serializable>,
    #     "correct": <bool>
    #   }
    samples = []
    with open(args.file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.num_samples > 0 and i >= args.num_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(
                dict(
                    output=obj.get("output", ""),
                    ground_truth=obj.get("ground_truth", None),
                    correct=bool(obj.get("correct", False)),
                )
            )

    if not samples:
        print("No samples loaded from the provided file.", file=sys.stderr, flush=True)
        raise SystemExit(1)

    print(f"[__main__] Loaded {len(samples)} samples.", flush=True)

    # ---- Tokenizer: use real HF tokenizer; use .encode() / .decode() ----
    print(f"[__main__] Loading tokenizer: {args.tokenizer_path}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print("[__main__] Tokenizer ready.", flush=True)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
        print(f"[__main__] pad_token_id not set; using {pad_token_id}", flush=True)

    # ---- Build padded tensors for REAL DataProto ----
    # We'll use empty prompts and map each sample["output"] to responses.
    prompt_ids_list = []
    response_ids_list = []
    ground_truth_list = []
    data_source_list = []

    for s in samples:
        prompt_ids = tokenizer.encode("", add_special_tokens=False)
        response_ids = tokenizer.encode(s["output"] or "", add_special_tokens=False)
        prompt_ids_list.append(prompt_ids)
        response_ids_list.append(response_ids)
        ground_truth_list.append({"ground_truth": s["ground_truth"]})
        data_source_list.append(args.data_source)

    B = len(samples)
    max_p = max((len(p) for p in prompt_ids_list), default=0)
    max_r = max((len(r) for r in response_ids_list), default=1)  # ensure >=1 to avoid zero-width reward tensor

    def pad_to(x, L, pad):
        return x + [pad] * (L - len(x))

    # Build [B, max_p], [B, max_r]
    prompts_tensor = torch.tensor([pad_to(p, max_p, pad_token_id) for p in prompt_ids_list], dtype=torch.long)
    responses_tensor = torch.tensor([pad_to(r, max_r, pad_token_id) for r in response_ids_list], dtype=torch.long)

    # Build attention_mask [B, max_p+max_r]: 1 for real tokens; 0 for padding
    attn = torch.zeros((B, max_p + max_r), dtype=torch.long)
    for i in range(B):
        lp = len(prompt_ids_list[i])
        lr = len(response_ids_list[i])
        if lp > 0:
            attn[i, :lp] = 1
        if lr > 0:
            attn[i, max_p:max_p + lr] = 1  # response section starts after max_p

    # NOTE: In `process_item`, for a single item we do:
    #   prompt_length = prompt_ids.shape[-1]  -> equals max_p
    #   valid_prompt_length = attention_mask[:prompt_length].sum()
    #   valid_response_length = attention_mask[prompt_length:].sum()
    # So we placed response bits starting from index max_p in attention_mask.

    # Pack non_tensor_batch as object arrays (DataProto will convert)
    data_source_arr = np.array(data_source_list, dtype=object)
    reward_model_arr = np.array(ground_truth_list, dtype=object)

    tensors = {
        "prompts": prompts_tensor,           # [B, max_p]
        "responses": responses_tensor,       # [B, max_r]
        "attention_mask": attn,              # [B, max_p + max_r]
    }
    non_tensors = {
        "data_source": data_source_arr,      # [B], each is str
        "reward_model": reward_model_arr,    # [B], each is {"ground_truth": ...}
    }

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    # ---- Request configuration (mirrors example client) ----
    request_config = dict(
        parallel_format_error_v2_reward_enabled=True,
        treat_no_parallel_as_format_error=True,
        parallel_format_error_v2_allow_nonempty_whitespace=True,
        acceleration_ratio_reward=0.1,
        acceleration_ratio_reward_factor=100,
        acceleration_ratio_clip_max=0.2,
        parallel_format_error_v2_skip_conclusion_check=True,
        allow_immediate_stop=True,
    )

    # ---- Instantiate RewardManagerWithServer, which auto-starts the server via subprocess ----
    rm = RewardManagerWithServer(
        tokenizer=tokenizer,  # provides .decode and .name_or_path
        num_examine=args.num_examine,
        config=request_config,
        tokenizer_path=args.tokenizer_path, # Explicitly pass CLI arg; if not provided, __init__ will use tokenizer.name_or_path
        server_host=args.host,
        server_port=args.port,
        workers=args.workers,
        server_timeout_s=args.server_timeout,
        auto_start_server=True
    )

    # Optional: set env for nicer debug filenames (not required)
    os.environ.setdefault("EXPERIMENT_NAME", "JSONL")
    os.environ.setdefault("GLOBAL_STEP", "0")
    os.environ.setdefault("SEQUENCE_DUMP_DIR", "./sequence_dumps")

    try:
        print("[__main__] Calling RewardManagerWithServer...", flush=True)
        out = rm(data_proto, return_dict=True, correctness_as_reward=bool(args.correctness_as_reward))
        # Pretty print, but convert tensors to lists for readability
        printable = {
            "reward_tensor": {
                "main_reward_tensor": out["reward_tensor"]["main_reward_tensor"][:2, -10:].tolist(),
                "secondary_reward_tensor": out["reward_tensor"]["secondary_reward_tensor"][:2, -10:].tolist(),
            },
            "reward_extra_info": {k: v[:2] for k, v in out["reward_extra_info"].items()},
        }
        print("\n=== RewardManagerWithServer output (dict) ===", flush=True)
        print(json.dumps(printable, indent=2), flush=True)

        print("[__main__] Mean reward:", np.mean(out["reward_extra_info"]["correct"]))
    finally:
        # Ensure server is cleaned up
        print("[__main__] Cleaning up reward server...", flush=True)
        del rm
