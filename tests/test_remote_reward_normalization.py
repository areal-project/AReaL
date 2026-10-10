from unittest.mock import Mock

import pytest

from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine


@pytest.mark.parametrize("engine_cls", [RemoteSGLangEngine, RemotevLLMEngine])
@pytest.mark.parametrize("method", ["submit", "rollout_batch", "prepare_batch"])
@pytest.mark.parametrize("use_std", [True, False])
def test_remote_engine_forwards_reward_normalization_mode(engine_cls, method, use_std):
    engine = engine_cls.__new__(engine_cls)
    engine._engine = Mock()
    payload_key = "dataloader" if method == "prepare_batch" else "data"
    kwargs = {
        payload_key: [],
        "workflow": "example.workflow",
        "reward_normalization": True,
        "reward_normalization_use_std": use_std,
    }

    result = getattr(engine, method)(**kwargs)

    delegated = getattr(engine._engine, method)
    assert result is delegated.return_value
    assert delegated.call_args.kwargs["reward_normalization"] is True
    assert delegated.call_args.kwargs["reward_normalization_use_std"] is use_std
