# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This script references code from Adaptive Parallel Reasoning (APR)/TinyRL's utils.py https://github.com/Parallel-Reasoning/APR/blob/main/tinyrl/utils.py and SGLang's test_utils.py https://github.com/sgl-project/sglang/blob/5d087891c93a6b66f0fd48b82fcf0a479d3e6ca5/python/sglang/test/test_utils.py#L545

The original script as well as the part from the original script used in this script are under Apache License 2.0 https://github.com/Parallel-Reasoning/APR/blob/main/LICENSE and https://github.com/sgl-project/sglang/blob/main/LICENSE
"""

import trl
import torch
import subprocess
import time
import requests
import os
from typing import List, Optional
from torch.utils.data import SequentialSampler
from sglang.srt.utils import kill_process_tree

def popen_launch_server(
    model: str,
    base_url: str,
    timeout: float,
    model_name: str = "model",
    api_key: Optional[str] = None,
    other_args: list[str] = (),
    env: Optional[dict] = None,
    return_stdout_stderr: Optional[tuple] = None,
    skip_actual_launch: bool = False,
    use_os_system: bool = False,
    wait_before_check: int = 0,
):
    _, host, port = base_url.split(":")
    host = host[2:]

    command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--served-model-name",
        model_name,
        *other_args,
    ]

    if api_key:
        command += ["--api-key", api_key]

    print(f"Launching server with command: {' '.join(command)}")

    if skip_actual_launch:
        process = None
    else:
        if use_os_system:
            command_str = " ".join(command) + " &"
            print(f"Executing command: {command_str}")
            os.system(command_str)
            # Servers launched with os.system do not return a process object and are not terminated automatically.
            process = None
        else:
            if return_stdout_stderr:
                process = subprocess.Popen(
                    command,
                    stdout=return_stdout_stderr[0],
                    stderr=return_stdout_stderr[1],
                    env=env,
                    text=True,
                )
            else:
                process = subprocess.Popen(
                    command, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    env=env
                )

    if wait_before_check > 0:
        print(f"Waiting for {wait_before_check} seconds before checking server status...")
        time.sleep(wait_before_check)

    start_time = time.time()
    with requests.Session() as session:
        while time.time() - start_time < timeout:
            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {api_key}",
                }
                response = session.get(
                    f"{base_url}/health_generate",
                    headers=headers,
                )
                if response.status_code == 200:
                    return process
            except requests.RequestException:
                pass
            time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")

def terminate_process(process):
    kill_process_tree(process.pid)

def add_and_init_special_tokens(model, tokenizer, new_special_tokens: Optional[List[str]] = None):
    """
    Adds new special tokens to the tokenizer and initializes their embeddings.
    """
    if new_special_tokens is None:
        new_special_tokens = [
            "<Think>", "</Think>", "<Parallel>", "</Parallel>",
            "<Outlines>", "</Outlines>", "<Outline>", "</Outline>",
            "<Trial>", "</Trial>", "<Subtask>", "</Subtask>",
            "<Thread>", "</Thread>", "<Conclusion>", "</Conclusion>"
        ]
    
    tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=64)

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()

    for tok in new_special_tokens:
        base_word = tok.strip("<>")
        base_ids = tokenizer(base_word, add_special_tokens=False).input_ids
        
        if all(i != tokenizer.unk_token_id for i in base_ids):
            avg_embed = embed(torch.tensor(base_ids, device=model.device)).mean(dim=0)
            special_id = tokenizer.convert_tokens_to_ids(tok)
            embed.weight.data[special_id] = avg_embed
            
            if not tied and lm_head.weight.shape == embed.weight.shape:
                avg_lm_logits = lm_head.weight.data[base_ids].mean(dim=0)
                lm_head.weight.data[special_id] = avg_lm_logits.clone()
        else:
            valid_ids = [i for i in base_ids if i != tokenizer.unk_token_id]
            print(f"Warning: Failed to init {tok}, some base tokens are unknown. Using available tokens: {[tokenizer.convert_ids_to_tokens(i) for i in valid_ids]}")
            if valid_ids:
                avg_embed = embed(torch.tensor(valid_ids, device=model.device)).mean(dim=0)
                special_id = tokenizer.convert_tokens_to_ids(tok)
                embed.weight.data[special_id] = avg_embed
                if not tied and lm_head.weight.shape == embed.weight.shape:
                    avg_lm_logits = lm_head.weight.data[valid_ids].mean(dim=0)
                    lm_head.weight.data[special_id] = avg_lm_logits.clone()




class SequentialSFTTrainer(trl.SFTTrainer):
    """
    Custom SFTTrainer that uses sequential sampling instead of random sampling
    """
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Override sampler method to use sequential sampling instead of random sampling"""
        if self.train_dataset is None or not hasattr(self.train_dataset, '__len__'):
            return None
        
        # If group_by_length is set, still use length-grouped sampler
        if self.args.group_by_length:
            return super()._get_train_sampler()
        else:
            # Use sequential sampler
            return SequentialSampler(self.train_dataset)
