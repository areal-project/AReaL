# SPDX-License-Identifier: Apache-2.0

import signal
from types import ModuleType

import pytest

from areal.utils import torch_npu_compat


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
