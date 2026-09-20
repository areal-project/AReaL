import pytest

from areal.reward.prm import BaseScorer, PRMConfig, PRMRunner


class ValidatingScorer(BaseScorer):
    name = "validating"

    def validate_prm_config(self, config, *, training_enabled):
        self.validated = training_enabled
        if config.advantage_shaping.mode != "additive":
            raise ValueError("additive required")

    async def evaluate(self, interaction, ctx):
        return 0.0


def test_prm_runner_invokes_scorer_validation():
    scorer = ValidatingScorer()
    runner = PRMRunner(PRMConfig(enabled=True, scorers=[scorer]))
    assert runner.supports_scorer_config_validation
    assert scorer.validated


def test_prm_runner_rejects_incompatible_scorer_config():
    config = PRMConfig(enabled=True, scorers=[ValidatingScorer()])
    config.advantage_shaping.mode = "gvpo"
    with pytest.raises(ValueError, match="additive required"):
        PRMRunner(config)
