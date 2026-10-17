import torch

from areal.experimental.dta.dta_engine import DTAEngine


class _DummyConfig:
    num_hidden_layers = 1
    num_key_value_heads = 1
    hidden_size = 8
    num_attention_heads = 1


class _FailIfCalledModel:
    def __call__(self, *args, **kwargs):
        raise AssertionError("Model must not be called when new_tokens is empty.")


def test_push_forward_only_empty_segment_does_not_call_model():
    engine = DTAEngine(
        model_config=_DummyConfig(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_seq_len=8,
        forward_only=True,
    )
    engine.model = _FailIfCalledModel()
    engine.returns = [None]

    empty_tokens = torch.empty(0, dtype=torch.long)
    engine.push_forward_only(
        empty_tokens,
        attach_list=[({"_sequence_batch_id": 0}, 0)],
    )

    assert engine.cur_len == 0
    assert isinstance(engine.returns[0], torch.Tensor)
    assert engine.returns[0].numel() == 0
