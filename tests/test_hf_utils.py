"""Tests for areal.utils.hf_utils.

Currently covers the multi-EOS resolution helpers:
get_eos_token_ids (union of EOS ids across generation_config + model
config, including the text_config VLM path and graceful degradation when the
config loaders fail) and resolve_stop_token_ids (tokenizer -> EOS list | None).

The two transformers loaders are monkeypatched, so these tests touch no network
and no real model files.
"""

from types import SimpleNamespace

import pytest
import transformers

from areal.utils.hf_utils import get_eos_token_ids, resolve_stop_token_ids


@pytest.fixture(autouse=True)
def _clear_eos_cache():
    """get_eos_token_ids is lru_cached; clear it so tests do not leak results."""
    get_eos_token_ids.cache_clear()
    yield
    get_eos_token_ids.cache_clear()


def _patch_configs(monkeypatch, *, gen_eos="__raise__", cfg=None):
    """Stub GenerationConfig/AutoConfig.from_pretrained for get_eos_token_ids.

    gen_eos: eos_token_id returned by the fake GenerationConfig, or the sentinel
        "__raise__" to make GenerationConfig.from_pretrained raise (config absent).
    cfg: object returned by the fake AutoConfig (use SimpleNamespace), or None to
        make AutoConfig.from_pretrained raise.
    """

    def fake_gen(_path, *args, **kwargs):
        if gen_eos == "__raise__":
            raise OSError("no generation_config.json")
        return SimpleNamespace(eos_token_id=gen_eos)

    def fake_auto(_path, *args, **kwargs):
        if cfg is None:
            raise OSError("no config.json")
        return cfg

    monkeypatch.setattr(
        transformers.GenerationConfig, "from_pretrained", staticmethod(fake_gen)
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", staticmethod(fake_auto)
    )


# --- get_eos_token_ids ------------------------------------------------------


def test_get_eos_token_ids_returns_single_int_from_generation_config(monkeypatch):
    """A scalar int eos_token_id in generation_config yields a 1-tuple."""
    _patch_configs(monkeypatch, gen_eos=2)
    assert get_eos_token_ids("model-a") == (2,)


def test_get_eos_token_ids_returns_sorted_unique_from_list(monkeypatch):
    """A list eos_token_id is deduplicated and sorted."""
    _patch_configs(monkeypatch, gen_eos=[50, 1, 106, 1])
    assert get_eos_token_ids("model-b") == (1, 50, 106)


def test_get_eos_token_ids_unions_generation_config_and_model_config(monkeypatch):
    """EOS ids from both generation_config and model config are unioned."""
    _patch_configs(monkeypatch, gen_eos=[1, 106], cfg=SimpleNamespace(eos_token_id=50))
    assert get_eos_token_ids("model-c") == (1, 50, 106)


def test_get_eos_token_ids_reads_text_config_for_vlm(monkeypatch):
    """Nested text_config.eos_token_id (VLM configs) is included."""
    cfg = SimpleNamespace(
        eos_token_id=None, text_config=SimpleNamespace(eos_token_id=[7, 8])
    )
    _patch_configs(monkeypatch, gen_eos="__raise__", cfg=cfg)
    assert get_eos_token_ids("model-vlm") == (7, 8)


def test_get_eos_token_ids_ignores_non_int_values(monkeypatch):
    """Non-int entries (e.g. None in a list) are skipped, not coerced."""
    _patch_configs(
        monkeypatch, gen_eos=[1, None, "x"], cfg=SimpleNamespace(eos_token_id=None)
    )
    assert get_eos_token_ids("model-d") == (1,)


def test_get_eos_token_ids_returns_empty_when_both_loaders_raise(monkeypatch):
    """Both config sources failing degrades to () instead of propagating.

    Guards the catch-all behaviour: malformed/gated/offline configs must not
    crash the generation path that builds a ModelResponse.
    """
    _patch_configs(monkeypatch, gen_eos="__raise__", cfg=None)
    assert get_eos_token_ids("missing-model") == ()


# --- resolve_stop_token_ids -------------------------------------------------


def test_resolve_stop_token_ids_none_tokenizer_returns_none():
    """A None tokenizer resolves to None."""
    assert resolve_stop_token_ids(None) is None


def test_resolve_stop_token_ids_empty_name_or_path_returns_none():
    """A tokenizer without a usable name_or_path resolves to None."""
    assert resolve_stop_token_ids(SimpleNamespace(name_or_path="")) is None


def test_resolve_stop_token_ids_returns_list_when_ids_found(monkeypatch):
    """A resolvable path returns the EOS ids as a list."""
    _patch_configs(monkeypatch, gen_eos=[1, 106])
    tokenizer = SimpleNamespace(name_or_path="model-e")
    assert resolve_stop_token_ids(tokenizer) == [1, 106]


def test_resolve_stop_token_ids_returns_none_when_no_ids(monkeypatch):
    """A resolvable path that yields no EOS ids resolves to None, not []."""
    _patch_configs(monkeypatch, gen_eos="__raise__", cfg=None)
    tokenizer = SimpleNamespace(name_or_path="missing-model")
    assert resolve_stop_token_ids(tokenizer) is None
