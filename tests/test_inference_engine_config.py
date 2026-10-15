# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from areal.api.cli_args import InferenceEngineConfig


def test_proxy_backend_scope_defaults_to_node() -> None:
    assert InferenceEngineConfig().proxy_backend_scope == "node"


@pytest.mark.parametrize("scope", ["node", "global"])
def test_proxy_backend_scope_survives_structured_config_override(scope: str) -> None:
    config = OmegaConf.merge(
        OmegaConf.structured(InferenceEngineConfig),
        OmegaConf.from_dotlist(
            [
                "backend=sglang:d1",
                "routing_strategy=least_active_groups",
                f"proxy_backend_scope={scope}",
            ]
        ),
    )

    assert OmegaConf.to_object(config).proxy_backend_scope == scope


def test_unknown_proxy_backend_scope_raises_value_error() -> None:
    with pytest.raises(ValueError, match="proxy_backend_scope must be"):
        InferenceEngineConfig(proxy_backend_scope="rack")


@pytest.mark.parametrize(
    ("version", "strategy"),
    [("v1", "round_robin"), ("v2", "least_active_groups")],
)
def test_global_proxy_scope_requires_session_routing(
    version: str, strategy: str
) -> None:
    with pytest.raises(ValueError, match="proxy_backend_scope='global' requires"):
        InferenceEngineConfig(
            _version=version,
            routing_strategy=strategy,
            proxy_backend_scope="global",
        )
