import sys
from types import SimpleNamespace

import pytest

from areal.api.cli_args import (
    BaseExperimentConfig,
    WandBConfig,
    WandBSystemMetricsConfig,
)
from areal.infra.rpc.serialization import serialize_value
from areal.utils.stats_logger import resolve_wandb_run_id
from areal.utils.wandb_system_metrics import (
    finish_worker_wandb_system_metrics,
    init_worker_wandb_system_metrics,
    prepare_wandb_run_identity,
    register_worker_wandb_system_metrics_hooks,
    worker_system_metrics_enabled,
)


def _make_config(tmp_path, *, roles=None, gpu_device_ids=None):
    config = BaseExperimentConfig(
        experiment_name="exp",
        trial_name="trial",
        total_train_epochs=1,
    )
    config.stats_logger.experiment_name = "exp"
    config.stats_logger.trial_name = "trial"
    config.stats_logger.fileroot = str(tmp_path)
    config.stats_logger.wandb = WandBConfig(
        mode="shared",
        project="proj",
        entity="entity",
        id_suffix="timestamp",
        system_metrics=WandBSystemMetricsConfig(
            enabled=True,
            roles=roles,
            gpu_device_ids=gpu_device_ids,
        ),
    )
    return config


def test_worker_system_metrics_requires_shared_mode():
    with pytest.raises(ValueError, match="requires stats_logger.wandb.mode='shared'"):
        WandBConfig(
            mode="online",
            system_metrics=WandBSystemMetricsConfig(enabled=True),
        )


def test_worker_system_metrics_rejects_empty_roles():
    with pytest.raises(ValueError, match="must be null or a non-empty list"):
        WandBSystemMetricsConfig(roles=[])


def test_worker_system_metrics_rejects_negative_gpu_ids():
    with pytest.raises(ValueError, match="must contain non-negative integers"):
        WandBSystemMetricsConfig(gpu_device_ids=[0, -1])


def test_worker_system_metrics_normalizes_iterables():
    cfg = WandBSystemMetricsConfig(roles=("actor", "rollout"), gpu_device_ids=(0, 1))
    assert cfg.roles == ["actor", "rollout"]
    assert cfg.gpu_device_ids == [0, 1]


def test_worker_system_metrics_default_roles_are_independent_lists():
    cfg = WandBSystemMetricsConfig()
    other = WandBSystemMetricsConfig()

    assert cfg.roles == ["actor", "rollout", "critic", "ref", "teacher"]
    assert isinstance(cfg.roles, list)

    cfg.roles.append("reward")
    assert other.roles == ["actor", "rollout", "critic", "ref", "teacher"]


def test_timestamp_run_id_is_pinned_before_workers_are_configured(
    monkeypatch, tmp_path
):
    config = _make_config(tmp_path)
    timestamps = iter(["2026_05_14_00_00_01", "2026_05_14_00_00_02"])
    monkeypatch.setattr(
        "areal.utils.stats_logger.time.strftime",
        lambda _: next(timestamps),
    )

    prepare_wandb_run_identity(config)

    assert config.stats_logger.wandb.id_suffix == "2026_05_14_00_00_01"
    run_id = resolve_wandb_run_id(config.stats_logger)
    assert run_id == "exp_trial_2026_05_14_00_00_01"
    assert resolve_wandb_run_id(config.stats_logger) == run_id


def test_run_id_is_left_alone_when_system_metrics_disabled(tmp_path):
    config = _make_config(tmp_path)
    config.stats_logger.wandb.system_metrics.enabled = False

    prepare_wandb_run_identity(config)

    assert config.stats_logger.wandb.id_suffix == "timestamp"


def test_worker_system_metrics_respects_role_filter(tmp_path):
    config = _make_config(tmp_path, roles=["actor"])

    assert worker_system_metrics_enabled(config, "actor")
    assert not worker_system_metrics_enabled(config, "rollout")


def test_worker_system_metrics_roles_none_skips_service_roles(tmp_path):
    config = _make_config(tmp_path, roles=None)

    assert worker_system_metrics_enabled(config, "actor")
    assert not worker_system_metrics_enabled(config, "actor-data")


def test_worker_wandb_init_uses_non_primary_shared_settings(monkeypatch, tmp_path):
    config = _make_config(tmp_path, roles=["actor"], gpu_device_ids=[0, 1])
    config.stats_logger.wandb.id_suffix = "fixed"
    calls = []
    run_finishes = []

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRun:
        def finish(self):
            run_finishes.append(True)

    def fail_global_finish():
        raise AssertionError("worker cleanup must finish the owned run handle")

    fake_wandb = SimpleNamespace(
        Settings=FakeSettings,
        init=lambda **kwargs: calls.append(kwargs) or FakeRun(),
        finish=fail_global_finish,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    assert init_worker_wandb_system_metrics(config, role="actor", rank=3)
    assert len(calls) == 1

    call = calls[0]
    assert call["mode"] == "shared"
    assert call["id"] == "exp_trial_fixed"
    assert call["settings"].kwargs == {
        "mode": "shared",
        "x_primary": False,
        "x_label": "actor-3",
        "x_update_finish_state": False,
        "x_stats_gpu_device_ids": [0, 1],
    }

    finish_worker_wandb_system_metrics()
    assert run_finishes == [True]


def test_configure_hook_labels_the_run_with_the_guard_role(monkeypatch, tmp_path):
    config = _make_config(tmp_path, roles=["actor"])
    config.stats_logger.wandb.id_suffix = "fixed"
    calls = []

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeState:
        def __init__(self, role):
            self.role = role
            self.configure_hooks = []
            self.cleanup_hooks = []

        def register_configure_hook(self, hook):
            self.configure_hooks.append(hook)

        def register_cleanup_hook(self, hook):
            self.cleanup_hooks.append(hook)

    fake_wandb = SimpleNamespace(
        Settings=FakeSettings,
        init=lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(finish=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    state = FakeState(role="actor")
    register_worker_wandb_system_metrics_hooks(state)
    (configure_hook,) = state.configure_hooks

    # The role comes from the guard state, not from the /configure payload.
    result = configure_hook({"config": serialize_value(config), "rank": 2})

    assert result == {"wandb_system_metrics": "enabled"}
    assert calls[0]["settings"].kwargs["x_label"] == "actor-2"

    for cleanup_hook in state.cleanup_hooks:
        cleanup_hook()


def test_worker_wandb_init_failure_does_not_crash(monkeypatch, tmp_path):
    config = _make_config(tmp_path, roles=["actor"])
    config.stats_logger.wandb.id_suffix = "fixed"

    class FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def boom(**_kwargs):
        raise RuntimeError("wandb backend unavailable")

    fake_wandb = SimpleNamespace(Settings=FakeSettings, init=boom, finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    assert not init_worker_wandb_system_metrics(config, role="actor", rank=0)
    finish_worker_wandb_system_metrics()
