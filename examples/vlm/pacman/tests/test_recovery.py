# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.mark.parametrize("curriculum", [1, 2])
@pytest.mark.parametrize("overlay", ["", "_awex_colocate"])
def test_recipe_recovery_preserves_optimizer_and_saves_synchronously(
    monkeypatch, tmp_path, curriculum, overlay
):
    for name in (
        "AREAL_ROOT",
        "MAAPACMAN_PACMAN_ROOT",
        "PACMAN_ARTIFACT_ROOT",
        "CURRICULUM1_CHECKPOINT",
    ):
        monkeypatch.setenv(name, str(tmp_path))
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1])
    ):
        config = compose(config_name=f"curriculum{curriculum}{overlay}")
    assert config.recover.mode == "auto"
    assert config.recover.no_save_optim is False
    assert config.recover.no_load_optim is False
    assert config.recover.freq_steps == 1
    assert config.actor.megatron.async_save is False
    assert config.recover.fileroot == config.cluster.fileroot
    assert config.recover.trial_name == config.trial_name
    if overlay:
        assert config.actor.offload is False
