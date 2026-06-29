# SPDX-License-Identifier: Apache-2.0

"""Tests for SGLang multi-node CLI generation."""

from __future__ import annotations

from unittest.mock import patch

from areal.api.cli_args import SGLangConfig


class TestSGLangMultiNode:
    """Mirror of TestVLLMMultiNode for the SGLang backend."""

    def _build_args(self, **kwargs):
        """Helper that patches sglang version checks away."""
        defaults = dict(
            sglang_config=SGLangConfig(model_path="test-model"),
            tp_size=8,
            base_gpu_id=0,
        )
        defaults.update(kwargs)
        with (
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.pkg_version.is_version_less", return_value=False),
        ):
            return SGLangConfig.build_args(**defaults)

    def _build_cmd(self, **kwargs):
        """Helper that patches sglang version checks away."""
        defaults = dict(
            sglang_config=SGLangConfig(model_path="test-model"),
            tp_size=8,
            base_gpu_id=0,
        )
        defaults.update(kwargs)
        with (
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.pkg_version.is_version_less", return_value=False),
        ):
            return SGLangConfig.build_cmd(**defaults)

    def test_build_args_single_node_defaults(self):
        """Single-node (default) should have nnodes=1, node_rank=0."""
        args = self._build_args()
        assert args["nnodes"] == 1
        assert args["node_rank"] == 0
        assert args.get("dist_init_addr") is None

    def test_build_args_multi_node_head(self):
        """Head node (rank 0) with n_nodes > 1 should set nnodes and dist_init_addr."""
        args = self._build_args(
            tp_size=16,
            n_nodes=2,
            node_rank=0,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert args["dist_init_addr"] == "10.0.0.1:29500"

    def test_build_args_multi_node_worker(self):
        """Worker node (rank > 0) should set nnodes and node_rank."""
        args = self._build_args(
            tp_size=16,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 1
        assert args["dist_init_addr"] == "10.0.0.1:29500"

    def test_build_args_multi_node_no_dist_init_addr(self):
        """Multi-node without dist_init_addr should have dist_init_addr=None."""
        args = self._build_args(
            tp_size=16,
            n_nodes=2,
            node_rank=0,
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert args.get("dist_init_addr") is None

    def test_build_cmd_multi_node_produces_flags(self):
        """build_cmd with multi-node should produce CLI flags for nnodes and node-rank."""
        cmd = self._build_cmd(
            tp_size=16,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        cmd_str = " ".join(cmd)
        assert "--nnodes" in cmd_str
        assert "--node-rank" in cmd_str
        assert "--dist-init-addr" in cmd_str
