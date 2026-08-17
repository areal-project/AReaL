from types import SimpleNamespace

from areal.trainer.rl_trainer import _set_default_agent_engine_max_tokens


def test_role_context_sets_unconfigured_agent_proxy_limit():
    config = SimpleNamespace(agent=SimpleNamespace(engine_max_tokens=None))
    _set_default_agent_engine_max_tokens(config, 40_000)
    assert config.agent.engine_max_tokens == 40_000

    eval_config = SimpleNamespace(agent=SimpleNamespace(engine_max_tokens=None))
    _set_default_agent_engine_max_tokens(eval_config, 262_144)
    assert eval_config.agent.engine_max_tokens == 262_144


def test_explicit_agent_proxy_limit_is_preserved():
    config = SimpleNamespace(agent=SimpleNamespace(engine_max_tokens=65_536))
    _set_default_agent_engine_max_tokens(config, 262_144)
    assert config.agent.engine_max_tokens == 65_536


def test_missing_agent_or_generation_limit_is_ignored():
    no_agent = SimpleNamespace(agent=None)
    _set_default_agent_engine_max_tokens(no_agent, 40_000)

    no_limit = SimpleNamespace(agent=SimpleNamespace(engine_max_tokens=None))
    _set_default_agent_engine_max_tokens(no_limit, None)
    assert no_limit.agent.engine_max_tokens is None
