import pytest

from areal.v2.inference_service.sglang.launch_server import (
    normalize_alloc_conf_for_inference,
    physical_base_from_cvd,
)


def test_normalize_alloc_conf_flips_expandable_segments_off():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    normalize_alloc_conf_for_inference(env)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_normalize_alloc_conf_preserves_other_tokens():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True"}
    normalize_alloc_conf_for_inference(env)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == (
        "max_split_size_mb:128,expandable_segments:False"
    )


def test_normalize_alloc_conf_leaves_unset_env_untouched():
    env = {}
    normalize_alloc_conf_for_inference(env)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env

    env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}
    normalize_alloc_conf_for_inference(env)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_physical_base_from_cvd_still_validates():
    with pytest.raises(ValueError):
        physical_base_from_cvd("0,2,3,4")
