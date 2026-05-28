import pytest

from areal.engine.fsdp_utils.attn_impl import (
    BUILTIN_ATTN_IMPLS,
    get_attn_impl_validation_error,
    is_valid_attn_impl,
)
from areal.tools.profile_fsdp import parse_args


@pytest.mark.parametrize(
    "attn_impl",
    [
        *BUILTIN_ATTN_IMPLS,
        "kernels-community/flash-attn",
        "kernels-community/flash-attn@main",
        "kernels-community/flash-attn:flash_attn_varlen_func",
        "kernels-community/flash-attn@main:flash_attn_varlen_func",
        "flash_attention_2|kernels-community/flash-attn@main:flash_attn_varlen_func",
    ],
)
def test_is_valid_attn_impl_accepts_builtin_and_kernel_repo_ids(attn_impl):
    assert is_valid_attn_impl(attn_impl)


@pytest.mark.parametrize(
    "attn_impl",
    [
        "kernels-community",
        "kernels-community/flash-attn/extra",
        "kernels-community/flash-attn:entry:extra",
        "kernels-community/",
    ],
)
def test_is_valid_attn_impl_rejects_invalid_values(attn_impl):
    assert not is_valid_attn_impl(attn_impl)


def test_get_attn_impl_validation_error_includes_invalid_value():
    assert "got 'bad/impl:too:many'" in get_attn_impl_validation_error(
        "bad/impl:too:many"
    )
    assert "optionally prefixed with wrapper|" in get_attn_impl_validation_error(
        "bad/impl:too:many"
    )


def test_profile_fsdp_parse_args_accepts_valid_kernel_repo_id():
    args = parse_args(["--attn-impl", "kernels-community/flash-attn@main"])

    assert args.attn_impl == "kernels-community/flash-attn@main"


def test_profile_fsdp_parse_args_accepts_wrapper_prefixed_kernel_repo_id():
    args = parse_args(
        [
            "--attn-impl",
            "flash_attention_2|kernels-community/flash-attn@main:flash_attn_varlen_func",
        ]
    )

    assert (
        args.attn_impl
        == "flash_attention_2|kernels-community/flash-attn@main:flash_attn_varlen_func"
    )


def test_profile_fsdp_parse_args_rejects_invalid_kernel_repo_id():
    with pytest.raises(SystemExit, match="2"):
        parse_args(["--attn-impl", "kernels-community/flash-attn:entry:extra"])
