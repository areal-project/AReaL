"""Tests for vLLM multi-node CLI generation."""

from __future__ import annotations

from areal.api.cli_args import vLLMConfig


class TestVLLMMultiNode:
    def test_build_args_single_node_no_extra_flags(self):
        """Single-node (default) should not add nnodes/node_rank/headless."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(cfg, tp_size=8, pp_size=1)
        assert "nnodes" not in args
        assert "node_rank" not in args
        assert "headless" not in args
        assert "master_addr" not in args
        assert "master_port" not in args

    def test_build_args_multi_node_head(self):
        """Head node (rank 0) with n_nodes > 1 should add nnodes/node_rank but NOT headless."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=0,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert "headless" not in args
        assert args["master_addr"] == "10.0.0.1"
        assert args["master_port"] == "29500"

    def test_build_args_multi_node_worker(self):
        """Worker node (rank > 0) should add headless=True."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 1
        assert args["headless"] is True
        assert args["master_addr"] == "10.0.0.1"
        assert args["master_port"] == "29500"

    def test_build_args_multi_node_no_dist_init_addr(self):
        """Multi-node without dist_init_addr should not add master_addr/master_port."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=0,
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert "master_addr" not in args
        assert "master_port" not in args

    def test_build_cmd_multi_node_produces_flags(self):
        """build_cmd with multi-node should produce CLI flags for nnodes and node-rank."""
        cfg = vLLMConfig(model="test-model")
        cmd = vLLMConfig.build_cmd(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        cmd_str = " ".join(cmd)
        assert "--nnodes" in cmd_str
        assert "--node-rank" in cmd_str
        assert "--headless" in cmd_str
        assert "--master-addr" in cmd_str
        assert "--master-port" in cmd_str
