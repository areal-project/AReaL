# SPDX-License-Identifier: Apache-2.0

from areal.infra.scheduler.colocation import colocated_train_guard_fork_env
from areal.v2.inference_service.sglang.launch_server import (
    normalize_alloc_conf_for_inference,
)


def test_fork_env_pins_gpu_and_strips_tms_preload():
    base_env = {
        "SOME_FLAG": "1",
        "LD_PRELOAD": "/x/torch_memory_saver_hook_mode_preload.abi3.so",
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }

    env = colocated_train_guard_fork_env(base_env, gpu_slot=5)

    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["SOME_FLAG"] == "1"
    assert env["LD_PRELOAD"] == ""
    assert env["TMS_INIT_ENABLE"] == "0"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "0"
    assert base_env["LD_PRELOAD"].endswith(".so")


def test_fork_env_overrides_inherited_tms_even_when_absent_from_base():
    env = colocated_train_guard_fork_env({}, gpu_slot=0)

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["LD_PRELOAD"] == ""
    assert env["TMS_INIT_ENABLE"] == "0"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "0"


def test_normalize_alloc_conf_preserves_other_tokens():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True"}
    normalize_alloc_conf_for_inference(env, awex_colocate=True)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == (
        "max_split_size_mb:128,expandable_segments:False"
    )


def test_normalize_alloc_conf_leaves_unset_env_untouched():
    env = {}
    normalize_alloc_conf_for_inference(env, awex_colocate=True)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env

    env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    normalize_alloc_conf_for_inference(env, awex_colocate=True)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_normalize_alloc_conf_leaves_separation_env_untouched():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    normalize_alloc_conf_for_inference(env, awex_colocate=False)

    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
