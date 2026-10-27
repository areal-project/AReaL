# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import MegatronEngineConfig, TrainEngineConfig


def test_vision_freezing_cli_overrides_survive_structured_config():
    config = OmegaConf.structured(
        TrainEngineConfig(
            backend="megatron:d1",
            experiment_name="vision-freezing",
            trial_name="test",
        )
    )

    parsed = OmegaConf.to_object(
        OmegaConf.merge(
            config,
            OmegaConf.from_dotlist(
                [
                    "megatron.bridge_type=megatron-bridge",
                    "megatron.freeze_vision_model=true",
                    "megatron.freeze_vision_projection=true",
                ]
            ),
        )
    )

    assert parsed.megatron.freeze_vision_model is True
    assert parsed.megatron.freeze_vision_projection is True


@pytest.mark.parametrize("bridge_type", ["mbridge", "mcore-bridge"])
@pytest.mark.parametrize("option", ["freeze_vision_model", "freeze_vision_projection"])
def test_vision_freezing_unsupported_bridge_raises(bridge_type, option):
    with pytest.raises(ValueError, match="require bridge_type='megatron-bridge'"):
        MegatronEngineConfig(bridge_type=bridge_type, **{option: True})


@pytest.mark.parametrize("freeze_encoder", [False, True])
@pytest.mark.parametrize("freeze_projection", [False, True])
def test_qwen35_native_freeze_keeps_frozen_weights_out_of_optimizer(
    freeze_encoder, freeze_projection
):
    # Exercise the installed provider's actual freeze contract without creating
    # a distributed VLM. In particular, encoder-only freezing re-enables mergers.
    module = pytest.importorskip(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model"
    )
    model = torch.nn.Module()
    model.language_model = torch.nn.Linear(2, 2)
    model.vision_model = torch.nn.Module()
    model.vision_model.encoder = torch.nn.Linear(2, 2)
    model.vision_model.merger = torch.nn.Linear(2, 2)
    model.vision_model.decoder = torch.nn.Module()
    model.vision_model.decoder.deepstack_merger_list = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2)]
    )
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.25)

    module.Qwen3VLModel.freeze(
        model,
        freeze_language_model=False,
        freeze_vision_model=freeze_encoder,
        freeze_vision_projection=freeze_projection,
    )
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    hidden = model.vision_model.encoder(torch.ones(1, 2))
    visual = model.vision_model.merger(hidden)
    visual = visual + model.vision_model.decoder.deepstack_merger_list[0](hidden)
    model.language_model(visual).square().mean().backward()
    optimizer.step()

    for name, parameter in model.named_parameters():
        expected_frozen = (
            freeze_encoder
            if name.startswith("vision_model.encoder.")
            else freeze_projection
            if name.startswith("vision_model.")
            else False
        )
        assert parameter.requires_grad is not expected_frozen
        if expected_frozen:
            assert parameter.grad is None
            assert all(
                parameter is not candidate
                for group in optimizer.param_groups
                for candidate in group["params"]
            )
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        else:
            assert parameter.grad is not None
            assert not torch.equal(parameter, before[name])
