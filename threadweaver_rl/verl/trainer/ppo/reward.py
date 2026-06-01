# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import importlib.util
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    function_name = reward_fn_config.get("name")
    assert function_name is not None

    module = sys.modules.get("custom_module", None)
    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location("custom_module", file_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_module"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    return partial(_call_with_kwargs, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(
    data: DataProto,
    reward_fn: AbstractRewardManager,
    **kwargs: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True, **kwargs)
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        raise

    reward_tensor = None
    reward_tensor_dict: dict[str, torch.Tensor] | None = None
    reward_extra_infos_dict = {}

    if isinstance(reward_result, dict):
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        reward_tensor = reward_result.get("reward_tensor")
        if reward_tensor is None:
            reward_tensor_dict = {
                k: v
                for k, v in reward_result.items()
                if isinstance(v, (torch.Tensor, float, int, list, tuple))
            }
    else:
        reward_tensor = reward_result

    if reward_tensor is not None and not isinstance(reward_tensor, dict):
        reward_tensor_dict = {
            "main_reward_tensor": reward_tensor
            if isinstance(reward_tensor, torch.Tensor)
            else torch.as_tensor(reward_tensor)
        }
    elif isinstance(reward_tensor, dict):
        reward_tensor_dict = {
            key: value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            for key, value in reward_tensor.items()
        }

    if reward_tensor_dict is None or not reward_tensor_dict:
        raise ValueError("Reward function returned no usable reward tensor.")

    keys = list(reward_tensor_dict.keys())
    first_tensor = reward_tensor_dict[keys[0]]
    if not isinstance(first_tensor, torch.Tensor):
        first_tensor = torch.as_tensor(first_tensor)
        reward_tensor_dict[keys[0]] = first_tensor
    for key in keys[1:]:
        tensor = reward_tensor_dict[key]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
            reward_tensor_dict[key] = tensor
        if tensor.shape != first_tensor.shape:
            raise ValueError(
                f"Reward tensor '{key}' shape {tensor.shape} does not match '{keys[0]}' shape {first_tensor.shape}."
            )

    reward_tensor_dict = {
        key: tensor if tensor.device == first_tensor.device else tensor.to(first_tensor.device)
        for key, tensor in reward_tensor_dict.items()
    }

    if "main_reward_tensor" not in reward_tensor_dict:
        reward_tensor_dict = {"main_reward_tensor": first_tensor, **reward_tensor_dict}
        reward_tensor_dict.setdefault("secondary_reward_tensor", torch.zeros_like(first_tensor))
    elif "secondary_reward_tensor" not in reward_tensor_dict:
        reward_tensor_dict["secondary_reward_tensor"] = torch.zeros_like(first_tensor)

    return reward_tensor_dict, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(
    data: DataProto,
    config: DictConfig | None = None,
    tokenizer: Any | None = None,
    reward_fn: AbstractRewardManager | None = None,
    **kwargs: Any,
):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn, **kwargs)
