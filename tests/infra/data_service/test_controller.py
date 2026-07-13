from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from areal.api.cli_args import (
    PPOConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SFTConfig,
)
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.controller.controller import DataController


def _make_mock_aiohttp(status=200, json_data=None, text_data=""):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data if json_data is not None else {})
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_session)
    return mock_cls, mock_session


class TestDataServiceConfig:
    def test_default_values(self):
        cfg = DataServiceConfig()

        assert cfg.num_workers == 1
        assert cfg.setup_timeout == 120.0
        assert isinstance(cfg.scheduling_strategy, SchedulingStrategy)
        assert cfg.scheduling_strategy.type == SchedulingStrategyType.separation

    def test_custom_values(self):
        cfg = DataServiceConfig(
            num_workers=8,
            setup_timeout=300.0,
        )

        assert cfg.num_workers == 8
        assert cfg.setup_timeout == 300.0

    def test_scheduling_strategy_colocation(self):
        cfg = DataServiceConfig(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            ),
        )

        assert cfg.scheduling_strategy.type == SchedulingStrategyType.colocation
        assert cfg.scheduling_strategy.target == "rollout"
        assert cfg.scheduling_strategy.fork is True

    def test_scheduling_strategy_separation(self):
        cfg = DataServiceConfig(
            scheduling_strategy=SchedulingStrategy(type="separation"),
        )

        assert cfg.scheduling_strategy.type == "separation"
        assert cfg.scheduling_strategy.target is None

    def test_config_in_base_experiment_config(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="rl")
        cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert isinstance(cfg, DataServiceConfig)

    def test_from_dataset_config_uses_num_dataset_workers(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy",
            type="rl",
            num_workers=5,
            num_dataset_workers=7,
        )
        cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert cfg.num_workers == 7
        assert cfg.dataloader_num_workers == 5

    def test_config_in_ppo_config(self):
        cfg = PPOConfig(experiment_name="exp", trial_name="trial")
        ds_cfg = DataServiceConfig.from_dataset_config(cfg.train_dataset)
        assert isinstance(ds_cfg, DataServiceConfig)

    def test_config_in_sft_config(self):
        cfg = SFTConfig(experiment_name="exp", trial_name="trial")
        ds_cfg = DataServiceConfig.from_dataset_config(cfg.train_dataset)
        assert isinstance(ds_cfg, DataServiceConfig)

    def test_config_in_rw_config(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="rl")
        cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert isinstance(cfg, DataServiceConfig)


class TestDataControllerInit:
    def test_init_stores_config(self):
        cfg = DataServiceConfig()
        scheduler = MagicMock()

        controller = DataController(cfg, scheduler)

        assert controller.config is cfg

    def test_init_stores_scheduler(self):
        cfg = DataServiceConfig()
        scheduler = MagicMock()

        controller = DataController(cfg, scheduler)

        assert controller.scheduler is scheduler

    def test_init_empty_state(self):
        controller = DataController(DataServiceConfig(), MagicMock())

        assert controller.workers == []
        assert controller._gateway_addr == ""
        assert controller._datasets == {}


class TestDataControllerGatewayPost:
    def test_gateway_post_sends_bearer_auth(self):
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        mock_cls, mock_session = _make_mock_aiohttp(status=200, json_data={"ok": True})

        with patch(
            "areal.infra.data_service.controller.controller.aiohttp.ClientSession",
            mock_cls,
        ):
            result = controller._gateway_post("/v1/test", "api-key", {"x": 1})

        assert result == {"ok": True}
        _, kwargs = mock_session.post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer api-key"}

    def test_gateway_post_sends_json_payload(self):
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        mock_cls, mock_session = _make_mock_aiohttp(status=200, json_data={"ok": True})

        with patch(
            "areal.infra.data_service.controller.controller.aiohttp.ClientSession",
            mock_cls,
        ):
            controller._gateway_post("/v1/test", "api-key", {"payload": "value"})

        _, kwargs = mock_session.post.call_args
        assert kwargs["json"] == {"payload": "value"}

    def test_gateway_post_raises_on_error(self):
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        mock_cls, _ = _make_mock_aiohttp(status=500, text_data="boom")

        with patch(
            "areal.infra.data_service.controller.controller.aiohttp.ClientSession",
            mock_cls,
        ):
            try:
                controller._gateway_post("/v1/test", "api-key", {})
            except RuntimeError as exc:
                assert "returned 500" in str(exc)
            else:
                raise AssertionError("Expected RuntimeError for gateway error response")


class TestDataControllerRegisterDataset:
    def test_register_returns_dataset_metadata(self):
        controller = DataController(DataServiceConfig(), MagicMock())

        payload = {
            "api_key": "ds-key",
            "dataset_id": "test-ds",
            "dataset_size": 32,
            "num_workers": 4,
        }

        with patch(
            "areal.infra.utils.concurrent.run_async_task",
            return_value=payload,
        ) as mock_run:
            result = controller.register_dataset(
                dataset_id="test-ds",
                dataset_path="dummy",
                dataset_type="rl",
                drop_last=True,
            )

        assert mock_run.called
        assert result["api_key"] == "ds-key"
        assert result["dataset_id"] == "test-ds"
        assert result["total_samples"] == 32
        assert result["num_workers"] == 4
        assert controller._datasets["ds-key"]["dataset_id"] == "test-ds"

    def test_register_stores_drop_last_flag(self):
        controller = DataController(DataServiceConfig(), MagicMock())

        payload = {
            "api_key": "ds-key",
            "dataset_id": "test-ds",
            "dataset_size": 30,
            "num_workers": 4,
        }
        with patch(
            "areal.infra.utils.concurrent.run_async_task",
            return_value=payload,
        ):
            controller.register_dataset(
                dataset_id="test-ds",
                dataset_path="dummy",
                dataset_type="rl",
                drop_last=False,
            )
        assert controller._datasets["ds-key"]["drop_last"] is False

    def test_unregister_removes_local_dataset_cache(self):
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._datasets["key-1"] = {"dataset_id": "a"}
        controller._datasets["key-2"] = {"dataset_id": "b"}

        with patch("areal.infra.utils.concurrent.run_async_task"):
            controller.unregister_dataset("a")

        assert "key-1" not in controller._datasets
        assert "key-2" in controller._datasets
