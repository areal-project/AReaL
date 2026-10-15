# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from areal.api.cli_args import InferenceEngineConfig


@pytest.mark.parametrize("scope", ["node", "global"])
def test_load_balance_names_configure_existing_routing(scope: str) -> None:
    config = OmegaConf.merge(
        OmegaConf.structured(InferenceEngineConfig),
        OmegaConf.from_dotlist(
            [
                "backend=sglang:d1",
                "load_balance_policy=least_groups",
                f"load_balance_scope={scope}",
            ]
        ),
    )
    resolved = OmegaConf.to_object(config)

    assert resolved.routing_strategy == "least_active_groups"
    assert resolved.proxy_backend_scope == scope
    # Persisted effective configs contain both spellings; loading them again
    # must preserve the same behavior rather than report an alias conflict.
    reloaded = OmegaConf.to_object(OmegaConf.structured(resolved))
    assert reloaded == resolved


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"load_balance_policy": "least_requests"}, "load_balance_policy must be"),
        ({"load_balance_scope": "rack"}, "load_balance_scope must be"),
        (
            {"load_balance_policy": "least_groups", "_version": "v2"},
            "load_balance_policy requires",
        ),
        (
            {"load_balance_scope": "node", "_version": "v2"},
            "load_balance_scope requires",
        ),
        ({"load_balance_scope": "global"}, "load_balance_scope='global' requires"),
        (
            {
                "load_balance_policy": "round_robin",
                "routing_strategy": "least_active_groups",
            },
            "load_balance_policy conflicts",
        ),
        (
            {"load_balance_scope": "node", "proxy_backend_scope": "global"},
            "load_balance_scope conflicts",
        ),
    ],
)
def test_invalid_load_balance_names_fail_clearly(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InferenceEngineConfig(**kwargs)


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
