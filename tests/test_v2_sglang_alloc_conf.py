from areal.v2.inference_service.sglang.launch_server import (
    normalize_alloc_conf_for_inference,
)


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
