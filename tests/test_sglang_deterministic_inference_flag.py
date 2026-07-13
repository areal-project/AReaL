from areal.api.cli_args import SGLangConfig


def _build_cmd(monkeypatch, **config_kwargs):
    # SGLangConfig.build_args guards on the `sglang` package being installed to check
    # its version (cli_args.py: `pkg_version.is_version_greater_or_equal("sglang", ...)`).
    # `sglang` is a GPU-only Linux package, unavailable on this machine and orthogonal
    # to the config-passthrough logic under test, so stub the version check rather than
    # bypass build_args/build_cmd_from_args (the real entry points SGLang launches use).
    monkeypatch.setattr(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
        lambda *a, **kw: True,
    )
    config = SGLangConfig(model_path="dummy", **config_kwargs)
    return SGLangConfig.build_cmd(
        sglang_config=config,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr="127.0.0.1:12345",
    )


def test_enable_deterministic_inference_flag_passed_through_when_true(monkeypatch):
    """Required for GenerationHyperparameters.sampling_seed to be honored by SGLang
    (SGLang gates per-request sampling_seed on this server flag)."""
    cmd = _build_cmd(monkeypatch, enable_deterministic_inference=True)

    assert "--enable-deterministic-inference" in cmd


def test_enable_deterministic_inference_flag_omitted_by_default(monkeypatch):
    """Default (False) must not appear on the command line, so existing SGLang
    launches are byte-for-byte unaffected."""
    cmd = _build_cmd(monkeypatch)

    assert "--enable-deterministic-inference" not in cmd
