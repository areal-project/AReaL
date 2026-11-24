# SPDX-License-Identifier: Apache-2.0

"""Backend validation matrix for score centering on the Megatron actor.

These cases only touch ``PPOActorConfig.__post_init__``: no ``megatron`` import,
no process group, so they run in the standard CPU CI environment.
"""

import pytest

from areal.api.cli_args import MegatronEngineConfig, PPOActorConfig

BASE_CONFIG = dict(experiment_name="x", trial_name="y", path="p")


def _actor_config(**overrides) -> PPOActorConfig:
    """Build a PPOActorConfig from the minimal required fields plus overrides."""
    return PPOActorConfig(**BASE_CONFIG, **overrides)


@pytest.mark.parametrize(
    "backend",
    [
        "megatron:d1p1t1",
        "megatron:d2p1t1",  # data parallel
        "megatron:d1p1t2",  # tensor parallel
        "megatron:d2p1t2",  # dp x tp
        "megatron:d2p2t2",  # + pipeline parallel
        "megatron:d2p1t2c2",  # + context parallel
        "megatron:(attn:d2p1t1c1|ffn:d1p1t1e2)",  # + expert parallel (MoE)
    ],
)
def test_sc_accepts_parallel_axis_when_megatron_backend(backend):
    """Test that every Megatron parallel axis is accepted when SC is enabled."""
    config = _actor_config(backend=backend, use_score_centering=True)

    assert config.backend == backend
    assert config.use_score_centering is True


def test_sc_accepts_fsdp_backend_when_enabled():
    """Test that the FSDP path keeps working unchanged."""
    config = _actor_config(backend="fsdp:d2p1t1", use_score_centering=True)

    assert config.use_score_centering is True


def test_sc_rejects_mtp_training_when_megatron_backend():
    """Test that Megatron MTP training is rejected because it scores extra rows."""
    with pytest.raises(ValueError, match="MTP"):
        _actor_config(
            backend="megatron:d2p1t1",
            use_score_centering=True,
            megatron=MegatronEngineConfig(
                bridge_type="megatron-bridge", enable_mtp_training=True
            ),
        )


def test_sc_rejects_chunked_logits_when_megatron_backend():
    """Test that the chunked LM head is rejected because it hides full logits."""
    with pytest.raises(ValueError, match="chunked_logits"):
        _actor_config(
            backend="megatron:d2p1t1",
            use_score_centering=True,
            megatron=MegatronEngineConfig(enable_chunked_logits=True),
        )


def test_sc_rejects_critic_when_enabled():
    """Test that a critic actor is rejected: SC scores sampler heads only."""
    with pytest.raises(ValueError, match="actor"):
        _actor_config(
            backend="megatron:d2p1t1", use_score_centering=True, is_critic=True
        )


def test_sc_accepts_pipeline_parallel_when_disabled():
    """Test that PP configs without SC are untouched by the new validation."""
    config = _actor_config(backend="megatron:d2p2t2", use_score_centering=False)

    assert config.use_score_centering is False


def test_sc_guard_ignores_mtp_when_disabled():
    """Test that MTP is only rejected while SC is enabled."""
    config = _actor_config(
        backend="megatron:d2p1t1",
        use_score_centering=False,
        megatron=MegatronEngineConfig(
            bridge_type="megatron-bridge", enable_mtp_training=True
        ),
    )

    assert config.use_score_centering is False
