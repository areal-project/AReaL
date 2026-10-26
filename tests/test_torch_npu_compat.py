# SPDX-License-Identifier: Apache-2.0

import signal
from types import ModuleType, SimpleNamespace

import pytest

from areal.utils import torch_npu_compat


def test_split_p2p_ops_uneven_peers_preserves_matching_rounds():
    """Each directed edge keeps its tensor order and agrees on round indices."""
    sends = SimpleNamespace(__name__="isend")
    recvs = SimpleNamespace(__name__="irecv")
    edges = {(0, 2): 5, (0, 3): 2, (1, 2): 1, (1, 3): 4, (2, 0): 3}
    schedules = {}
    for rank in range(4):
        ops = []
        for (src, dst), count in edges.items():
            for index in range(count):
                if rank == src:
                    ops.append(
                        SimpleNamespace(op=sends, peer=dst, tensor=(src, dst, index))
                    )
                if rank == dst:
                    ops.append(
                        SimpleNamespace(op=recvs, peer=src, tensor=(src, dst, index))
                    )
        rounds = torch_npu_compat._split_p2p_ops(ops)
        assert sum(map(len, rounds)) == len(ops)
        for index, batch in enumerate(rounds):
            keys = [(op.op.__name__, op.peer) for op in batch]
            assert len(keys) == len(set(keys))
            for op in batch:
                assert op.tensor[2] == index
                schedules[rank, op.tensor] = index
    for (src, dst), count in edges.items():
        for index in range(count):
            assert (
                schedules[src, (src, dst, index)] == schedules[dst, (src, dst, index)]
            )


def test_split_p2p_ops_unique_peers_keeps_one_batch():
    """Warmup and mixed send/receive pairs need no extra HCCL calls."""
    ops = [
        SimpleNamespace(op=SimpleNamespace(__name__=direction), peer=peer)
        for direction, peer in [("isend", 1), ("irecv", 1), ("isend", 2)]
    ]
    assert torch_npu_compat._split_p2p_ops(ops) == [ops]
    assert torch_npu_compat._split_p2p_ops([]) == []


def _mock_signal_state(monkeypatch):
    original_sigint = object()
    original_sigterm = object()
    handlers = {
        signal.SIGINT: original_sigint,
        signal.SIGTERM: original_sigterm,
    }

    monkeypatch.setattr(signal, "getsignal", handlers.__getitem__)

    def set_handler(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        return previous

    monkeypatch.setattr(signal, "signal", set_handler)
    return handlers, original_sigint, original_sigterm


def test_import_mindspeed_adaptor_restores_signal_handlers(monkeypatch):
    handlers, original_sigint, original_sigterm = _mock_signal_state(monkeypatch)
    torchair_handler = object()
    adaptor = ModuleType("megatron_adaptor")

    def import_module(name):
        assert name == "megatron_adaptor"
        handlers[signal.SIGINT] = torchair_handler
        handlers[signal.SIGTERM] = torchair_handler
        return adaptor

    monkeypatch.setattr(torch_npu_compat.importlib, "import_module", import_module)

    assert torch_npu_compat.import_mindspeed_adaptor() is adaptor
    assert handlers == {
        signal.SIGINT: original_sigint,
        signal.SIGTERM: original_sigterm,
    }


def test_import_mindspeed_adaptor_restores_handlers_after_error(monkeypatch):
    handlers, original_sigint, original_sigterm = _mock_signal_state(monkeypatch)
    torchair_handler = object()

    def import_module(name):
        assert name == "megatron_adaptor"
        handlers[signal.SIGINT] = torchair_handler
        handlers[signal.SIGTERM] = torchair_handler
        raise RuntimeError("adaptor import failed")

    monkeypatch.setattr(torch_npu_compat.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="adaptor import failed"):
        torch_npu_compat.import_mindspeed_adaptor()

    assert handlers == {
        signal.SIGINT: original_sigint,
        signal.SIGTERM: original_sigterm,
    }


def test_import_mindspeed_adaptor_skips_signal_calls_off_main_thread(monkeypatch):
    adaptor = ModuleType("megatron_adaptor")
    monkeypatch.setattr(torch_npu_compat.threading, "current_thread", lambda: object())
    monkeypatch.setattr(
        torch_npu_compat.importlib, "import_module", lambda name: adaptor
    )
    monkeypatch.setattr(
        torch_npu_compat.signal,
        "getsignal",
        lambda signum: pytest.fail("getsignal should not be called"),
    )
    monkeypatch.setattr(
        torch_npu_compat.signal,
        "signal",
        lambda signum, handler: pytest.fail("signal should not be called"),
    )

    assert torch_npu_compat.import_mindspeed_adaptor() is adaptor
