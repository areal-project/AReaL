from areal.utils import timeutil


def test_frequency_control_seconds_without_distributed_returns_false(monkeypatch):
    """Time checks should work before torch.distributed is initialized."""
    monkeypatch.setattr(timeutil.dist, "is_initialized", lambda: False)
    control = timeutil.FrequencyControl(frequency_seconds=60)

    result = control.check()

    assert result is False
