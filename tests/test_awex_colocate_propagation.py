"""Unit tests for AWEX colocate MetaServer address propagation.

Verifies that:
1. WeightUpdateMeta.from_awex() propagates meta_server_addr
2. SGLang launch_server() activates AWEX plugin entry when flags present
3. awex_run_scheduler_process always applies memory patch, registers plugin only with addr
4. init_colocate_weight_update skips start_meta_server when addr is provided

These tests require the full AReaL runtime (flask, torchdata, sglang, awex).
They are designed to run in the cluster container. Skipped if deps are missing.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Guard: skip entire module if critical deps are missing
pytest.importorskip("flask", reason="flask required (run in cluster container)")
pytest.importorskip("torchdata", reason="torchdata required (run in cluster container)")


class TestWeightUpdateMetaFromAwex:
    """WeightUpdateMeta.from_awex() must propagate meta_server_addr."""

    def test_from_awex_with_addr(self):
        from areal.api.io_struct import WeightUpdateMeta

        alloc = MagicMock()
        meta = WeightUpdateMeta.from_awex(alloc, meta_server_addr="10.0.0.1:8765")
        assert meta.type == "awex"
        assert meta.meta_server_addr == "10.0.0.1:8765"
        assert meta.alloc_mode is alloc

    def test_from_awex_without_addr(self):
        from areal.api.io_struct import WeightUpdateMeta

        alloc = MagicMock()
        meta = WeightUpdateMeta.from_awex(alloc)
        assert meta.type == "awex"
        assert meta.meta_server_addr is None
        assert meta.alloc_mode is alloc

    def test_from_awex_none_addr_explicit(self):
        from areal.api.io_struct import WeightUpdateMeta

        alloc = MagicMock()
        meta = WeightUpdateMeta.from_awex(alloc, meta_server_addr=None)
        assert meta.meta_server_addr is None


class TestSGLangLaunchServerAwexPlugin:
    """SGLangBackend.launch_server() activates AWEX plugin when flags present."""

    @patch("subprocess.Popen")
    def test_awex_colocate_mode_activates_plugin(self, mock_popen):
        from areal.engine.sglang_remote import SGLangBackend

        mock_popen.return_value = MagicMock()
        backend = SGLangBackend()

        server_args = {
            "awex_colocate_mode": True,
            "awex_meta_server_addr": "10.0.0.1:8765",
            "model_path": "/some/model",
            "tp_size": 4,
            "port": 30000,
        }

        with patch(
            "areal.api.cli_args.SGLangConfig.build_cmd_from_args",
            return_value=[
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                "/some/model",
            ],
        ):
            backend.launch_server(server_args)

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        env = call_args[1]["env"]

        assert "areal.engine.awex_sglang_plugin" in cmd
        assert "sglang.launch_server" not in cmd
        assert env.get("AWEX_META_SERVER_ADDR") == "10.0.0.1:8765"

    @patch("subprocess.Popen")
    def test_awex_colocate_mode_without_addr_still_activates(self, mock_popen):
        from areal.engine.sglang_remote import SGLangBackend

        mock_popen.return_value = MagicMock()
        backend = SGLangBackend()

        server_args = {
            "awex_colocate_mode": True,
            "model_path": "/some/model",
            "tp_size": 4,
            "port": 30000,
        }

        with (
            patch(
                "areal.api.cli_args.SGLangConfig.build_cmd_from_args",
                return_value=[
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    "/some/model",
                ],
            ),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("AWEX_META_SERVER_ADDR", None)
            backend.launch_server(server_args)

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert "areal.engine.awex_sglang_plugin" in cmd

    @patch.dict("os.environ", {"SLURM_LOCALID": "1"}, clear=False)
    @patch("subprocess.Popen")
    def test_colocate_slurm_localid_overrides_base_gpu_id(self, mock_popen):
        # Real colocate (controller injected _awex_gpus_per_server) + SLURM_LOCALID
        # present -> base_gpu_id recomputed from the node-slot id, overriding the
        # controller's (potentially colliding) fallback value.
        from areal.engine.sglang_remote import SGLangBackend

        mock_popen.return_value = MagicMock()
        backend = SGLangBackend()

        server_args = {
            "awex_colocate_mode": True,
            "_awex_gpus_per_server": 4,
            "base_gpu_id": 0,  # controller fallback (e.g. collided to 0)
            "model_path": "/some/model",
            "tp_size": 4,
            "port": 30000,
        }

        captured = {}

        def _fake_build(args):
            captured["base_gpu_id"] = args.get("base_gpu_id")
            captured["has_internal_key"] = "_awex_gpus_per_server" in args
            return [sys.executable, "-m", "sglang.launch_server", "--model-path", "/m"]

        with patch(
            "areal.api.cli_args.SGLangConfig.build_cmd_from_args",
            side_effect=_fake_build,
        ):
            backend.launch_server(server_args)

        # SLURM_LOCALID(1) * gpus_per_server(4) == 4
        assert captured["base_gpu_id"] == 4
        # internal key must be popped before building the SGLang CLI
        assert captured["has_internal_key"] is False

    @patch("subprocess.Popen")
    def test_colocate_no_slurm_localid_keeps_base_gpu_id(self, mock_popen):
        # Colocate signal present but SLURM_LOCALID unavailable (local/Ray) ->
        # keep the controller-provided fallback base_gpu_id unchanged.
        import os

        from areal.engine.sglang_remote import SGLangBackend

        mock_popen.return_value = MagicMock()
        backend = SGLangBackend()

        server_args = {
            "awex_colocate_mode": True,
            "_awex_gpus_per_server": 4,
            "base_gpu_id": 8,  # controller fallback must be preserved
            "model_path": "/some/model",
            "tp_size": 4,
            "port": 30000,
        }

        captured = {}

        def _fake_build(args):
            captured["base_gpu_id"] = args.get("base_gpu_id")
            return [sys.executable, "-m", "sglang.launch_server"]

        with (
            patch(
                "areal.api.cli_args.SGLangConfig.build_cmd_from_args",
                side_effect=_fake_build,
            ),
            patch.dict("os.environ", {}, clear=False),
        ):
            os.environ.pop("SLURM_LOCALID", None)
            backend.launch_server(server_args)

        assert captured["base_gpu_id"] == 8

    @patch.dict("os.environ", {"SLURM_LOCALID": "1"}, clear=False)
    @patch("subprocess.Popen")
    def test_separated_no_gpus_per_server_no_override(self, mock_popen):
        # Separated AWEX still sets awex_colocate_mode=True (rl_trainer keys it on
        # weight_update_mode), but the controller does NOT inject
        # _awex_gpus_per_server. base_gpu_id must stay untouched even with
        # SLURM_LOCALID set, otherwise CVD-isolated processes would go out of range.
        from areal.engine.sglang_remote import SGLangBackend

        mock_popen.return_value = MagicMock()
        backend = SGLangBackend()

        server_args = {
            "awex_colocate_mode": True,
            "base_gpu_id": 0,
            "model_path": "/some/model",
            "tp_size": 4,
            "port": 30000,
        }

        captured = {}

        def _fake_build(args):
            captured["base_gpu_id"] = args.get("base_gpu_id")
            captured["has_internal_key"] = "_awex_gpus_per_server" in args
            return [sys.executable, "-m", "sglang.launch_server"]

        with patch(
            "areal.api.cli_args.SGLangConfig.build_cmd_from_args",
            side_effect=_fake_build,
        ):
            backend.launch_server(server_args)

        assert captured["base_gpu_id"] == 0
        # separated never injects the internal key in the first place
        assert captured["has_internal_key"] is False

    @patch("subprocess.Popen")
    def test_no_awex_flags_uses_standard_entry(self, mock_popen):
        from areal.engine.sglang_remote import SGLangBackend

        mock_popen.return_value = MagicMock()
        backend = SGLangBackend()

        server_args = {
            "model_path": "/some/model",
            "tp_size": 4,
            "port": 30000,
        }

        with (
            patch(
                "areal.api.cli_args.SGLangConfig.build_cmd_from_args",
                return_value=[
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    "/some/model",
                ],
            ),
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("AWEX_META_SERVER_ADDR", None)
            backend.launch_server(server_args)

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert "sglang.launch_server" in cmd
        assert "areal.engine.awex_sglang_plugin" not in cmd


@pytest.mark.skipif(
    "awex" not in sys.modules
    and pytest.importorskip("awex", reason="awex not installed"),
    reason="awex package required",
)
class TestAwexRunSchedulerProcess:
    """awex_run_scheduler_process registers the plugin only when addr is set."""

    @patch.dict("os.environ", {"AWEX_META_SERVER_ADDR": "10.0.0.1:8765"})
    @patch("areal.engine.awex_sglang_plugin.register_awex_plugin")
    @patch("sglang.srt.managers.scheduler.run_scheduler_process")
    def test_with_addr_registers_plugin(self, mock_run, mock_register):
        from areal.engine.awex_sglang_plugin import awex_run_scheduler_process

        awex_run_scheduler_process()

        mock_register.assert_called_once()
        mock_run.assert_called_once()

    @patch("areal.engine.awex_sglang_plugin.register_awex_plugin")
    @patch("sglang.srt.managers.scheduler.run_scheduler_process")
    def test_without_addr_skips_registration(self, mock_run, mock_register):
        import os

        os.environ.pop("AWEX_META_SERVER_ADDR", None)

        from areal.engine.awex_sglang_plugin import awex_run_scheduler_process

        awex_run_scheduler_process()

        mock_register.assert_not_called()
        mock_run.assert_called_once()


@pytest.mark.skipif(
    "awex" not in sys.modules
    and pytest.importorskip("awex", reason="awex not installed"),
    reason="awex package required",
)
class TestAwexColocateInitSkipsDoubleStart:
    """init_colocate_weight_update should not start MetaServer when addr is provided."""

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("awex.meta.meta_server.start_meta_server")
    @patch("awex.meta.meta_server.MetaServerClient")
    def test_provided_addr_no_start(
        self, mock_client_cls, mock_start, mock_ws, mock_rank
    ):
        mock_client_cls.return_value = MagicMock()

        from areal.engine.awex_colocate_writer import AwexColocateWriter

        engine = MagicMock()
        adapter = AwexColocateWriter(engine)
        adapter.init_colocate_weight_update(meta_server_addr="10.0.0.1:8765")

        mock_start.assert_not_called()
        mock_client_cls.assert_called_once_with("10.0.0.1", 8765)

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("awex.meta.meta_server.start_meta_server", return_value=("0.0.0.0", 9999))
    @patch("awex.meta.meta_server.MetaServerClient")
    def test_no_addr_starts_server(
        self, mock_client_cls, mock_start, mock_ws, mock_rank
    ):
        import os

        os.environ.pop("AWEX_META_SERVER_ADDR", None)
        mock_client_cls.return_value = MagicMock()

        from areal.engine.awex_colocate_writer import AwexColocateWriter

        engine = MagicMock()
        adapter = AwexColocateWriter(engine)
        adapter.init_colocate_weight_update()

        mock_start.assert_called_once()
