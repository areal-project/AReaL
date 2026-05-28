# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch


def get_state_dict_from_repo_id_or_path(repo_id_or_path: str) -> dict:
    """
    Obtain a state dictionary from either a Hugging Face repo ID or a local path.

    Args:
        repo_id_or_path (str): Either a Hugging Face repo ID (e.g., 'username/model-name')
                              or a local path to a directory containing model weights.

    Returns:
        Dict: The combined state dictionary from all .safetensors and .bin files.
    """
    from safetensors.torch import load_file as safetensors_load

    state_dict = {}

    # Step 1: Identify if the input is a Hugging Face repo ID or local path
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        try:
            validate_repo_id(repo_id_or_path)
            is_hf_repo = True
        except HFValidationError:
            is_hf_repo = False
    except ImportError:
        is_hf_repo = False

    if is_hf_repo:
        from huggingface_hub import snapshot_download

        # Step 2: Download the repo if it's a Hugging Face repo ID
        local_path = snapshot_download(
            repo_id=repo_id_or_path,
        )
    else:
        # Assume it's a local path
        local_path = repo_id_or_path

    # Step 3: Load all .safetensors and .bin files
    file_paths_to_load = []
    if os.path.isdir(local_path):
        for filename in os.listdir(local_path):
            filepath = os.path.join(local_path, filename)
            if filename.endswith(".safetensors") or filename.endswith(".bin"):
                file_paths_to_load.append(filepath)
    elif os.path.isfile(local_path):
        file_paths_to_load.append(local_path)
    else:
        raise ValueError(
            f"Local path {local_path} does not exist or is not a valid path, "
            f"or {local_path} is a huggingface repo id but huggingface_hub is not installed."
        )

    def _load(filepath: str):
        if filepath.endswith(".safetensors"):
            state_dict = safetensors_load(filepath)
        elif filepath.endswith(".bin"):
            state_dict = torch.load(filepath, map_location="cpu")
        else:
            raise ValueError(f"{filepath} is not a torch bin or safetensor file.")
        return state_dict

    state_dict = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(
        max_workers=min(4, max(1, os.cpu_count() // 8))
    ) as executor:
        future_to_checkpoint = {
            executor.submit(_load, path): path for path in file_paths_to_load
        }

        for future in as_completed(future_to_checkpoint):
            path = future_to_checkpoint[future]
            try:
                sd = future.result()
                state_dict.update(sd)
            except Exception as e:
                raise RuntimeError(f"Error loading checkpoint from {path}: {e}")
    return state_dict


def is_existing_local_path(path: str) -> bool:
    try:
        path_obj = Path(path)
        return path_obj.exists() and (path_obj.is_file() or path_obj.is_dir())
    except (ValueError, OSError):
        return False
