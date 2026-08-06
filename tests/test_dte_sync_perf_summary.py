# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_parser_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "areal"
        / "tools"
        / "dte_sync_perf_summary.py"
    )
    spec = importlib.util.spec_from_file_location("dte_sync_perf_summary", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PARSER = _load_parser_module()
format_markdown_summary = _PARSER.format_markdown_summary
main = _PARSER.main
parse_sync_perf_logs = _PARSER.parse_sync_perf_logs


def test_parse_sync_perf_logs_summarizes_versions(tmp_path: Path) -> None:
    """Parse real-style DTE timing lines into per-version p50/min/max stats."""

    driver_log = tmp_path / "driver.out"
    driver_log.write_text(
        "\n".join(
            [
                "\x1b[37m20260708 WeightUpdateGateway INFO: Weight update completed for pair 'actor-rollout' v2 (61479.4ms)",
                "│ timeperf/update_weights              │  1.0329e+02 │ other │",
                "│ timeperf/train_step                  │  3.1464e+01 │ ppo_actor/update/perf/optimizer_step_time       │  3.1896e+00 │",
                "│ ppo_actor/update/perf/step_dirty_capture_time │  4.7627e+00 │ ppo_actor/update/perf/step_dirty_compare_time │  9.8212e+00 │",
                "│ ppo_actor/update/perf/step_dirty_pack_time    │  1.2696e+00 │ ppo_actor/update/perf/step_dirty_indices_time │  2.3491e+00 │",
                "│ ppo_actor/update/perf/dirty_bit_collect_time  │  1.5000e-03 │ ppo_actor/update/perf/dirty_bit_records │  2.0000e+00 │",
                "20260708 WeightUpdateGateway INFO: Weight update completed for pair 'actor-rollout' v3 (65345.0ms)",
                "│ timeperf/update_weights              │  1.0975e+02 │ other │",
            ]
        ),
        encoding="utf-8",
    )

    train_log = tmp_path / "train-worker.log"
    train_log.write_text(
        "\n".join(
            [
                "INFO: [dte-perf][train-delta] v2 detector=inversion mode=delta compute_masks_ms=27753.9 encode_ms=402.2 payload_mb=490.0 dense_mb=8037.5 total_ms=28156.5",
                "INFO: [dte-perf][train-delta] v2 detector=inversion mode=delta compute_masks_ms=29886.6 encode_ms=539.1 payload_mb=477.8 dense_mb=8037.5 total_ms=30426.1",
                "INFO: [dte-perf][inversion] v2 result=sparse reconstruct_mcore_ms=26422.4 convert_hf_ms=81.6 mask_loop_ms=1244.0 total_ms=27748.9",
                "INFO: [dte-perf][inversion] v2 result=sparse reconstruct_mcore_ms=28021.4 convert_hf_ms=99.4 mask_loop_ms=1755.2 total_ms=29877.3",
                "INFO: colocate delta v2 [inversion]: changed 81671733/4018747392 (2.03%) sparse=2073 dense_fallback=1 unchanged=665 payload=490.0MB vs dense=8037.5MB",
                "INFO: colocate delta v2 [inversion]: changed 79641041/4018747392 (1.98%) sparse=2083 dense_fallback=1 unchanged=655 payload=477.8MB vs dense=8037.5MB",
                "INFO: [dte-perf][train] v2 rank 8 sync_model_params_ms=1533.2 total_ms=1533.3",
                "INFO: [dte-perf][train] v2 rank 8 release_grad_ms=1073.6 total_ms=33187.4",
                "INFO: [dte-perf][train] v2 rank 8 group_payload_tensors_ms=880.0 total_ms=34123.4",
                "INFO: [dte-perf][train] v2 rank 8 serialize_ipc_payload_ms=690.0 total_ms=34813.4",
                "INFO: [dte-perf][train] v2 rank 8 wait_inference_done_and_mark_synced_ms=9688.9 total_ms=44502.3",
                "INFO: [dte-perf][train] v2 rank 8 cleanup_ms=1373.5 total_ms=45875.8",
                "INFO: [dte-perf][step-dirty] phase=before storage=cpu total_params=4 captured_params=2 skipped_by_cap=2 snapshot_mb=128.0 total_snapshot_mb=512.0 capture_ms=41.0",
                "INFO: [dte-perf][step-dirty] phase=after captured_params=2 captured_elements=64000000 changed_elements=1280000 changed_ratio=0.020000 snapshot_mb=128.0 bitset_mb=8.0 indices_elements=1280000 indices_mb=5.12 compare_ms=52.0 pack_ms=0.0 indices_ms=7.0",
                "INFO: [dte-perf][step-dirty] phase=before storage=cpu total_params=4 captured_params=2 skipped_by_cap=2 snapshot_mb=128.0 total_snapshot_mb=512.0 capture_ms=61.0",
                "INFO: [dte-perf][step-dirty] phase=after captured_params=2 captured_elements=64000000 changed_elements=640000 changed_ratio=0.010000 snapshot_mb=128.0 bitset_mb=8.0 indices_elements=640000 indices_mb=2.56 compare_ms=72.0 pack_ms=5.0 indices_ms=9.0",
                "INFO: [dte-perf][dirty-bit-provider] records=2 complete=1 collect_ms=1.5",
                "INFO: [dte-perf][dirty-bit-provider] records=0 complete=1 collect_ms=0.0",
            ]
        ),
        encoding="utf-8",
    )

    infer_log = tmp_path / "inf-server.log"
    infer_log.write_text(
        "\n".join(
            [
                "INFO: [dte-perf][infer] v2 rank 0 paired_train_rank=8 wait_train_offloaded_ms=33542.4 total_ms=33542.4",
                "INFO: [dte-perf][infer] v2 rank 0 paired_train_rank=8 resume_weights_ms=4680.5 total_ms=44122.0",
                "INFO: [dte-perf][infer] v2 rank 0 paired_train_rank=8 apply_decoded_delta_colocate_ms=5113.8 total_ms=49235.8",
                "INFO: [dte-perf][infer] v2 rank 1 paired_train_rank=9 wait_train_offloaded_ms=40005.1 total_ms=40005.1",
                "INFO: [dte-perf][infer] v2 rank 1 paired_train_rank=9 resume_weights_ms=11359.7 total_ms=51364.8",
                "INFO: [dte-perf][infer] v2 rank 1 paired_train_rank=9 apply_decoded_delta_colocate_ms=5158.7 total_ms=56523.5",
                "INFO: [dte-perf][infer] v1 rank 0 paired_train_rank=8 apply_full_colocate_ms=55509.6 total_ms=91995.7",
                "INFO: [dte-perf][infer] v1 rank 1 paired_train_rank=9 apply_full_colocate_ms=57701.3 total_ms=94188.3",
                "INFO: [dte-perf][infer] v3 rank 0 paired_train_rank=8 commit_empty_delta_ms=0.4 total_ms=16564.2",
                "INFO: [dte-perf][awex-recursive] rank=infer_0 step=1 rounds=3 total_ms=998.6 send_ops=896 recv_ops=896",
                "INFO: [dte-perf][awex-recursive] rank=infer_1 step=1 rounds=3 total_ms=111.9 send_ops=896 recv_ops=896",
                "INFO: [dte-perf][awex-chunk] task=infer_0-1 chunk=14/20 send_peers=7 recv_peers=7 clone_mb=672.0 total_ms=1077.5",
                "INFO: [dte-perf][awex-chunk] task=infer_1-1 chunk=14/20 send_peers=7 recv_peers=7 clone_mb=633.1 total_ms=1067.1",
                "[dte-perf][delta-build] rank=infer_0 step=2 phase=self ops=10 sparse_ops=8 dense_ops=0 empty_ops=2 sparse_groups=8 avg_ops_per_group=1.00 max_ops_per_group=1 sparse_input_nnz=12000000 sparse_output_nnz=4000000 sparse_zero_ops=0 first_pass_ms=1.0 dense_ms=0.0 sparse_remap_ms=100.0 dtype_cast_ms=0.0 total_ms=120.0",
                "[dte-perf][delta-build] rank=infer_0 step=2 phase=cross dtype=torch.bfloat16 ops=70 sparse_ops=56 dense_ops=0 empty_ops=14 sparse_groups=8 avg_ops_per_group=7.00 max_ops_per_group=7 sparse_input_nnz=100000000 sparse_output_nnz=180000000 sparse_zero_ops=0 first_pass_ms=2.0 dense_ms=0.0 sparse_remap_ms=300.0 dtype_cast_ms=0.0 total_ms=330.0",
                "[dte-perf][delta-build] rank=infer_1 step=2 phase=cross dtype=torch.bfloat16 ops=70 sparse_ops=49 dense_ops=0 empty_ops=21 sparse_groups=7 avg_ops_per_group=7.00 max_ops_per_group=7 sparse_input_nnz=200000000 sparse_output_nnz=300000000 sparse_zero_ops=1 first_pass_ms=3.0 dense_ms=0.0 sparse_remap_ms=500.0 dtype_cast_ms=0.0 total_ms=540.0",
            ]
        ),
        encoding="utf-8",
    )

    summary = parse_sync_perf_logs(driver_log, train_log, infer_log)

    assert summary["gateway_ms"]["2"] == 61479.4
    assert summary["trainer_update_s"]["2"] == 103.29
    assert summary["driver_stats"]["2"]["train_step_s"]["p50"] == 31.464
    assert summary["driver_stats"]["2"]["optimizer_step_s"]["p50"] == 3.1896
    assert summary["driver_stats"]["2"]["step_dirty_capture_s"]["p50"] == 4.7627
    assert summary["driver_stats"]["2"]["step_dirty_indices_s"]["p50"] == 2.3491
    assert summary["driver_stats"]["2"]["dirty_bit_collect_s"]["p50"] == 0.0015
    assert summary["driver_stats"]["2"]["dirty_bit_records"]["p50"] == 2.0
    assert summary["delta_summary"]["2"]["changed_pct"]["p50"] == 2.005
    assert summary["train_delta"]["2"]["compute_masks_ms"]["p50"] == 28820.25
    assert summary["inversion"]["2"]["reconstruct_mcore_ms"]["p50"] == 27221.9
    assert summary["train_stage"]["2"]["release_grad_ms"]["p50"] == 1073.6
    assert summary["train_stage"]["2"]["group_payload_tensors_ms"]["p50"] == 880.0
    assert summary["train_stage"]["2"]["serialize_ipc_payload_ms"]["p50"] == 690.0
    assert (
        summary["train_stage"]["2"]["wait_inference_done_and_mark_synced_ms"]["p50"]
        == 9688.9
    )
    assert summary["step_dirty"]["capture_ms"]["p50"] == 51.0
    assert summary["step_dirty"]["changed_ratio"]["p50"] == 0.015
    assert summary["step_dirty"]["indices_ms"]["p50"] == 8.0
    assert summary["step_dirty"]["indices_mb"]["p50"] == 3.84
    assert summary["step_dirty_by_round"]["1"]["capture_ms"]["p50"] == 41.0
    assert summary["step_dirty_by_round"]["1"]["changed_ratio"]["p50"] == 0.02
    assert summary["step_dirty_by_round"]["1"]["indices_ms"]["p50"] == 7.0
    assert summary["step_dirty_by_round"]["2"]["capture_ms"]["p50"] == 61.0
    assert summary["step_dirty_by_round"]["2"]["changed_ratio"]["p50"] == 0.01
    assert summary["step_dirty_by_round"]["2"]["indices_ms"]["p50"] == 9.0
    assert summary["dirty_bit_provider"]["records"]["p50"] == 1.0
    assert summary["dirty_bit_provider"]["complete"]["p50"] == 1.0
    assert summary["dirty_bit_provider"]["collect_ms"]["p50"] == 0.75
    assert summary["infer"]["1"]["apply_full_colocate_ms"]["p50"] == 56605.45
    assert summary["infer"]["2"]["apply_decoded_delta_colocate_ms"]["p50"] == 5136.25
    assert summary["infer"]["3"]["commit_empty_delta_ms"]["p50"] == 0.4
    assert summary["awex_recursive"]["1"]["total_ms"]["p50"] == 555.25
    assert summary["awex_chunk"]["1"]["total_ms"]["p50"] == 1072.3
    assert summary["awex_chunk"]["1"]["clone_mb"]["p50"] == 652.55
    assert summary["delta_build"]["2"]["self"]["sparse_remap_ms"]["p50"] == 100.0
    assert (
        summary["delta_build"]["2"]["cross"]["sparse_input_nnz"]["p50"] == 150000000.0
    )
    assert summary["delta_build"]["2"]["cross"]["sparse_remap_ms"]["p50"] == 400.0


def test_format_markdown_summary_contains_key_sections() -> None:
    """Render a compact markdown report from parsed summary dictionaries."""

    summary = {
        "gateway_ms": {"2": 61479.4},
        "trainer_update_s": {"2": 103.29},
        "delta_summary": {
            "2": {
                "changed_pct": {"count": 1, "p50": 2.04, "min": 2.04, "max": 2.04},
                "payload_mb": {"count": 1, "p50": 490.6, "min": 490.6, "max": 490.6},
            }
        },
        "train_delta": {
            "2": {
                "compute_masks_ms": {
                    "count": 1,
                    "p50": 28665.0,
                    "min": 28665.0,
                    "max": 28665.0,
                },
                "encode_ms": {"count": 1, "p50": 348.0, "min": 348.0, "max": 348.0},
            }
        },
        "inversion": {},
        "driver_stats": {
            "2": {
                "train_step_s": {
                    "count": 1,
                    "p50": 31.464,
                    "min": 31.464,
                    "max": 31.464,
                },
                "update_weights_s": {
                    "count": 1,
                    "p50": 103.29,
                    "min": 103.29,
                    "max": 103.29,
                },
                "optimizer_step_s": {
                    "count": 1,
                    "p50": 3.1896,
                    "min": 3.1896,
                    "max": 3.1896,
                },
                "step_dirty_capture_s": {
                    "count": 1,
                    "p50": 4.7627,
                    "min": 4.7627,
                    "max": 4.7627,
                },
                "step_dirty_compare_s": {
                    "count": 1,
                    "p50": 9.8212,
                    "min": 9.8212,
                    "max": 9.8212,
                },
                "step_dirty_pack_s": {
                    "count": 1,
                    "p50": 1.2696,
                    "min": 1.2696,
                    "max": 1.2696,
                },
                "step_dirty_indices_s": {
                    "count": 1,
                    "p50": 2.3491,
                    "min": 2.3491,
                    "max": 2.3491,
                },
            }
        },
        "step_dirty": {
            "capture_ms": {"count": 1, "p50": 41.0, "min": 41.0, "max": 41.0},
            "compare_ms": {"count": 1, "p50": 52.0, "min": 52.0, "max": 52.0},
            "pack_ms": {"count": 1, "p50": 0.0, "min": 0.0, "max": 0.0},
            "indices_ms": {"count": 1, "p50": 7.0, "min": 7.0, "max": 7.0},
            "snapshot_mb": {"count": 1, "p50": 128.0, "min": 128.0, "max": 128.0},
            "bitset_mb": {"count": 1, "p50": 8.0, "min": 8.0, "max": 8.0},
            "indices_mb": {"count": 1, "p50": 5.12, "min": 5.12, "max": 5.12},
            "captured_params": {"count": 1, "p50": 2.0, "min": 2.0, "max": 2.0},
            "changed_ratio": {"count": 1, "p50": 0.02, "min": 0.02, "max": 0.02},
        },
        "dirty_bit_provider": {
            "collect_ms": {"count": 1, "p50": 1.5, "min": 1.5, "max": 1.5},
            "records": {"count": 1, "p50": 2.0, "min": 2.0, "max": 2.0},
            "complete": {"count": 1, "p50": 1.0, "min": 1.0, "max": 1.0},
        },
        "train_stage": {
            "2": {
                "sync_model_params_ms": {
                    "count": 1,
                    "p50": 1533.2,
                    "min": 1533.2,
                    "max": 1533.2,
                },
                "delta_or_full_encode_ms": {
                    "count": 1,
                    "p50": 539.1,
                    "min": 539.1,
                    "max": 539.1,
                },
                "release_grad_ms": {
                    "count": 1,
                    "p50": 1073.6,
                    "min": 1073.6,
                    "max": 1073.6,
                },
                "group_payload_tensors_ms": {
                    "count": 1,
                    "p50": 880.0,
                    "min": 880.0,
                    "max": 880.0,
                },
                "release_weights_ms": {
                    "count": 1,
                    "p50": 1030.0,
                    "min": 1030.0,
                    "max": 1030.0,
                },
                "serialize_ipc_payload_ms": {
                    "count": 1,
                    "p50": 690.0,
                    "min": 690.0,
                    "max": 690.0,
                },
                "wait_inference_done_and_mark_synced_ms": {
                    "count": 1,
                    "p50": 9688.9,
                    "min": 9688.9,
                    "max": 9688.9,
                },
                "cleanup_ms": {
                    "count": 1,
                    "p50": 1373.5,
                    "min": 1373.5,
                    "max": 1373.5,
                },
            }
        },
        "step_dirty_by_round": {
            "1": {
                "capture_ms": {"count": 1, "p50": 41.0, "min": 41.0, "max": 41.0},
                "compare_ms": {"count": 1, "p50": 52.0, "min": 52.0, "max": 52.0},
                "pack_ms": {"count": 1, "p50": 0.0, "min": 0.0, "max": 0.0},
                "indices_ms": {"count": 1, "p50": 7.0, "min": 7.0, "max": 7.0},
                "snapshot_mb": {"count": 1, "p50": 128.0, "min": 128.0, "max": 128.0},
                "bitset_mb": {"count": 1, "p50": 8.0, "min": 8.0, "max": 8.0},
                "indices_mb": {"count": 1, "p50": 5.12, "min": 5.12, "max": 5.12},
                "changed_ratio": {"count": 1, "p50": 0.02, "min": 0.02, "max": 0.02},
            }
        },
        "infer": {
            "1": {
                "apply_full_colocate_ms": {
                    "count": 1,
                    "p50": 57240.0,
                    "min": 57240.0,
                    "max": 57240.0,
                }
            },
            "2": {
                "apply_decoded_delta_colocate_ms": {
                    "count": 1,
                    "p50": 5151.0,
                    "min": 5151.0,
                    "max": 5151.0,
                }
            },
            "3": {
                "commit_empty_delta_ms": {
                    "count": 1,
                    "p50": 0.4,
                    "min": 0.4,
                    "max": 0.4,
                }
            },
        },
        "awex_recursive": {
            "1": {
                "total_ms": {
                    "count": 1,
                    "p50": 998.6,
                    "min": 998.6,
                    "max": 998.6,
                },
                "send_ops": {
                    "count": 1,
                    "p50": 896.0,
                    "min": 896.0,
                    "max": 896.0,
                },
                "recv_ops": {
                    "count": 1,
                    "p50": 896.0,
                    "min": 896.0,
                    "max": 896.0,
                },
            }
        },
        "awex_chunk": {
            "1": {
                "total_ms": {
                    "count": 2,
                    "p50": 1072.3,
                    "min": 1067.1,
                    "max": 1077.5,
                },
                "clone_mb": {
                    "count": 2,
                    "p50": 652.55,
                    "min": 633.1,
                    "max": 672.0,
                },
                "send_peers": {
                    "count": 2,
                    "p50": 7.0,
                    "min": 7.0,
                    "max": 7.0,
                },
                "recv_peers": {
                    "count": 2,
                    "p50": 7.0,
                    "min": 7.0,
                    "max": 7.0,
                },
            }
        },
        "delta_build": {
            "2": {
                "cross": {
                    "sparse_input_nnz": {
                        "count": 1,
                        "p50": 100000000.0,
                        "min": 100000000.0,
                        "max": 100000000.0,
                    },
                    "sparse_output_nnz": {
                        "count": 1,
                        "p50": 180000000.0,
                        "min": 180000000.0,
                        "max": 180000000.0,
                    },
                    "sparse_ops": {
                        "count": 1,
                        "p50": 56.0,
                        "min": 56.0,
                        "max": 56.0,
                    },
                    "sparse_groups": {
                        "count": 1,
                        "p50": 8.0,
                        "min": 8.0,
                        "max": 8.0,
                    },
                    "avg_ops_per_group": {
                        "count": 1,
                        "p50": 7.0,
                        "min": 7.0,
                        "max": 7.0,
                    },
                    "sparse_remap_ms": {
                        "count": 1,
                        "p50": 300.0,
                        "min": 300.0,
                        "max": 300.0,
                    },
                    "total_ms": {
                        "count": 1,
                        "p50": 330.0,
                        "min": 330.0,
                        "max": 330.0,
                    },
                }
            }
        },
    }

    rendered = format_markdown_summary(summary)

    assert "# DTE Sync Perf Summary" in rendered
    assert "| v2 | 61.479s | 103.290s | 2.040% [2.040, 2.040] |" in rendered
    assert "28.665s [28.665, 28.665]" in rendered
    assert "## Trainer Stats" in rendered
    assert "3.190s [3.190, 3.190]" in rendered
    assert "## Optimizer Step Dirty Dry-Run" in rendered
    assert "## Dirty-Bit Provider" in rendered
    assert (
        "| 0.002s [0.002, 0.002] | 2.000 [2.000, 2.000] | 1.000 [1.000, 1.000] |"
        in rendered
    )
    assert "## Training Sender Stages" in rendered
    assert "9.689s [9.689, 9.689]" in rendered
    assert "## Optimizer Step Dirty Dry-Run By Round" in rendered
    assert "| 1 | 0.041s [0.041, 0.041] | 0.052s [0.052, 0.052] |" in rendered
    assert "## Inference Receiver" in rendered
    assert "57.240s [57.240, 57.240]" in rendered
    assert "0.000s [0.000, 0.000]" in rendered
    assert "## AWEX P2P" in rendered
    assert (
        "| v1 | 0.999s [0.999, 0.999] | 896.000 [896.000, 896.000] | 896.000 [896.000, 896.000] |"
        in rendered
    )
    assert "## Receiver Delta Build" in rendered
    assert "| v2 | cross | 100.0M [100.0, 100.0] | 180.0M [180.0, 180.0] |" in rendered


def test_main_json_outputs_summary(tmp_path: Path, capsys) -> None:
    """CLI JSON mode prints machine-readable summary data."""

    driver_log = tmp_path / "driver.out"
    driver_log.write_text(
        "Weight update completed for pair 'actor-rollout' v2 (61479.4ms)\n",
        encoding="utf-8",
    )

    exit_code = main(["--driver-log", str(driver_log), "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gateway_ms"]["2"] == 61479.4
