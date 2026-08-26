# SPDX-License-Identifier: Apache-2.0

"""Guarded first-use probe for workflows that reach the engine directly.

The proxy probes at initialization; workflows like VisionRLVRWorkflow call
agenerate() themselves, so the contract has to be verified lazily before the
first multimodal request instead of discovered during rollout traffic.
"""

import asyncio
import itertools

import pytest

from areal.utils import vision_canary


class _StubProcessor:
    """Stands in for an AutoProcessor; a new instance per workflow."""

    name_or_path = "model-a"


class _StubVisionPrompt:
    input_ids = [1, 2, 3]
    collapsed_input_ids = [1, 2]


def _stub_process_vision_prompt(processor, text, image_data):
    return _StubVisionPrompt()


def _patch_probe_deps(monkeypatch):
    """Stub the heavy helpers the probe imports lazily, at their source."""
    monkeypatch.setattr(
        "areal.experimental.openai.client._process_vision_prompt",
        _stub_process_vision_prompt,
    )
    monkeypatch.setattr(
        "areal.utils.hf_utils.apply_chat_template", lambda *a, **k: "text"
    )
    # Distinct per call, so a test can tell the probe images apart the way the
    # real encoder would.
    counter = itertools.count()
    monkeypatch.setattr(
        "areal.utils.image.image2base64", lambda *a: [f"b64-{next(counter)}"]
    )


@pytest.fixture(autouse=True)
def _clean_probe_state():
    vision_canary.reset_for_testing()
    yield
    vision_canary.reset_for_testing()


@pytest.mark.asyncio
async def test_text_only_skips_the_probe(monkeypatch):
    """Test that a text-only workflow does no multimodal work."""
    calls = []
    monkeypatch.setattr(
        vision_canary, "run_exact_token_canary", lambda *a: calls.append(a)
    )

    await vision_canary.ensure_exact_token_support(_InnerEngine([]), None, object())

    assert calls == []


@pytest.mark.asyncio
async def test_probe_runs_once_per_engine_and_processor(monkeypatch):
    """Test that later requests reuse the first probe's result."""
    calls = []

    async def fake(engine, processor, tokenizer):
        calls.append(engine)

    monkeypatch.setattr(vision_canary, "run_exact_token_canary", fake)
    engine, processor, tokenizer = _InnerEngine(["a:1"]), _StubProcessor(), object()

    for _ in range(3):
        await vision_canary.ensure_exact_token_support(engine, processor, tokenizer)

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_concurrent_first_requests_share_one_probe(monkeypatch):
    """Test that racing callers do not each probe the servers."""
    started = 0

    async def fake(engine, processor, tokenizer):
        nonlocal started
        started += 1
        await asyncio.sleep(0.05)

    monkeypatch.setattr(vision_canary, "run_exact_token_canary", fake)
    engine, processor, tokenizer = _InnerEngine(["a:1"]), _StubProcessor(), object()

    await asyncio.gather(
        *(
            vision_canary.ensure_exact_token_support(engine, processor, tokenizer)
            for _ in range(8)
        )
    )

    assert started == 1


@pytest.mark.asyncio
async def test_failure_rejects_every_waiter(monkeypatch):
    """Test that a failing probe fails all concurrent callers.

    Letting the losers proceed would run them against an unverified server.
    """

    async def fake(engine, processor, tokenizer):
        await asyncio.sleep(0.02)
        raise RuntimeError("server rejected the contract")

    monkeypatch.setattr(vision_canary, "run_exact_token_canary", fake)
    engine, processor, tokenizer = _InnerEngine(["a:1"]), _StubProcessor(), object()

    results = await asyncio.gather(
        *(
            vision_canary.ensure_exact_token_support(engine, processor, tokenizer)
            for _ in range(4)
        ),
        return_exceptions=True,
    )

    assert len(results) == 4
    assert all(isinstance(r, RuntimeError) for r in results)


@pytest.mark.asyncio
async def test_a_repaired_server_can_be_probed_again(monkeypatch):
    """Test that a failure is not cached forever."""
    attempts = 0

    async def fake(engine, processor, tokenizer):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("first attempt fails")

    monkeypatch.setattr(vision_canary, "run_exact_token_canary", fake)
    engine, processor, tokenizer = _InnerEngine(["a:1"]), _StubProcessor(), object()

    with pytest.raises(RuntimeError):
        await vision_canary.ensure_exact_token_support(engine, processor, tokenizer)
    await vision_canary.ensure_exact_token_support(engine, processor, tokenizer)

    assert attempts == 2


# ---------------------------------------------------------------------------
# Engine resolution and identity
# ---------------------------------------------------------------------------


class _InnerEngine:
    """A server with the exact-token validation installed.

    Rejects any request whose expected prompt is not the one it would produce,
    which is what the canary's negative control relies on.
    """

    def __init__(self, addresses, lora_name="", tokenizer_path="model-a"):
        self.addresses = list(addresses)
        self.rid_to_address = {}
        self.config = type(
            "Cfg", (), {"lora_name": lora_name, "tokenizer_path": tokenizer_path}
        )()
        self.requests = []

    async def agenerate(self, request):
        self.requests.append(request)
        if list(request.input_ids) != list(_StubVisionPrompt.input_ids):
            raise RuntimeError(
                "400, message='[areal-exact-token] Rendered prompt does not "
                "match expected_token_ids. expected_len=3 actual_len=3'"
            )
        return object()


class _SingleBackendEngine:
    """Mirrors InfBridge: one backend_addr, no round-robin, no rid pinning.

    The data proxy hands this shape to the canary, so probing it must not
    depend on ``addresses`` or ``rid_to_address``.
    """

    def __init__(self, backend_addr="http://bridge:8000", tokenizer_path="model-a"):
        self.backend_addr = backend_addr
        self.config = type(
            "Cfg", (), {"lora_name": "", "tokenizer_path": tokenizer_path}
        )()
        self.requests = []

    async def agenerate(self, request):
        self.requests.append(request)
        if list(request.input_ids) != list(_StubVisionPrompt.input_ids):
            raise RuntimeError(
                "400, message='[areal-exact-token] Rendered prompt does not "
                "match expected_token_ids. expected_len=3 actual_len=3'"
            )
        return object()


class _UnpatchedEngine(_InnerEngine):
    """A server missing the validation patch.

    pydantic drops unknown request fields, so expected_token_ids is ignored and
    everything is accepted -- including a prompt the server cannot reproduce.
    """

    async def agenerate(self, request):
        self.requests.append(request)
        return object()


class _WrapperEngine:
    """Mirrors RemotevLLMEngine: routing lives on an inner engine."""

    def __init__(self, inner):
        self._engine = inner
        self.config = inner.config


@pytest.mark.asyncio
async def test_probe_reaches_through_the_engine_wrapper(monkeypatch):
    """Test that the wrapper's inner engine is probed, not the wrapper.

    The public engines hold routing on ``_engine``; probing the wrapper would
    find no servers and report success without sending anything.
    """
    inner = _InnerEngine(["a:1", "b:2"])
    _patch_probe_deps(monkeypatch)

    await vision_canary.run_exact_token_canary(
        _WrapperEngine(inner), _StubProcessor(), object()
    )

    # Two addresses, each probed with a valid request then a negative control.
    assert len(inner.requests) == 4
    assert {r for r in inner.rid_to_address} == set()  # every pin cleaned up


@pytest.mark.asyncio
async def test_zero_addresses_is_an_error_not_a_pass(monkeypatch):
    """Test that probing nothing fails loudly.

    Reporting success after checking no servers is worse than not checking.
    """
    _patch_probe_deps(monkeypatch)

    with pytest.raises(RuntimeError, match="no server addresses"):
        await vision_canary.run_exact_token_canary(
            _InnerEngine([]), _StubProcessor(), object()
        )


@pytest.mark.asyncio
async def test_probe_selects_the_configured_lora_adapter(monkeypatch):
    """Test that the probe uses the deployment's adapter, not the default.

    GenerationHyperparameters defaults to "default_lora", which a LoRA
    deployment has not preloaded, so the canary would fail a healthy server.
    """
    inner = _InnerEngine(["a:1"], lora_name="my-adapter")
    _patch_probe_deps(monkeypatch)

    await vision_canary.run_exact_token_canary(inner, _StubProcessor(), object())

    assert inner.requests[0].gconfig.lora_name == "my-adapter"


@pytest.mark.asyncio
async def test_recreated_processors_share_one_probe(monkeypatch):
    """Test that a fresh processor per workflow does not re-probe.

    Workflows are re-resolved per submitted item and build a new processor each
    time, so identity-keyed caching would probe every server for every sample.
    """
    calls = []

    async def fake(engine, processor, tokenizer):
        calls.append(processor)

    monkeypatch.setattr(vision_canary, "run_exact_token_canary", fake)
    inner = _InnerEngine(["a:1"])

    for _ in range(4):
        await vision_canary.ensure_exact_token_support(
            inner,
            _StubProcessor(),
            object(),  # a new processor object each time
        )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_cancelled_waiter_does_not_break_the_shared_probe(monkeypatch):
    """Test that one cancelled caller does not corrupt the probe for others.

    Awaiting the shared future directly would let a cancelled waiter cancel it,
    so the owner raises InvalidStateError and a cancelled probe stays cached.
    """
    started = asyncio.Event()

    async def fake(engine, processor, tokenizer):
        started.set()
        await asyncio.sleep(0.15)

    monkeypatch.setattr(vision_canary, "run_exact_token_canary", fake)
    inner = _InnerEngine(["a:1"])
    processor = _StubProcessor()

    doomed = asyncio.create_task(
        vision_canary.ensure_exact_token_support(inner, processor, object())
    )
    await started.wait()
    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed

    # A later caller still observes a healthy, completed probe.
    await vision_canary.ensure_exact_token_support(inner, processor, object())


@pytest.mark.asyncio
async def test_canary_detects_a_server_that_ignores_expected_token_ids(monkeypatch):
    """Test that a server without the validation patch is caught.

    pydantic drops unknown fields silently, so such a server accepts
    expected_token_ids without enforcing it and a positive-only probe would
    report success while nothing was being validated.
    """
    _patch_probe_deps(monkeypatch)

    with pytest.raises(RuntimeError, match="not.*installed"):
        await vision_canary.run_exact_token_canary(
            _UnpatchedEngine(["a:1"]), _StubProcessor(), object()
        )


@pytest.mark.asyncio
async def test_the_control_is_invalid_and_carries_a_throwaway_image(monkeypatch):
    """Test the probe order and that the control uses a throwaway image.

    Positive first: a rejected request is rendered before it is refused, so
    leading with the control would strand its image in the server's cache and
    the valid request would then be sent by-reference against nothing.
    """
    _patch_probe_deps(monkeypatch)
    engine = _InnerEngine(["a:1"])

    await vision_canary.run_exact_token_canary(engine, _StubProcessor(), object())

    assert len(engine.requests) == 2
    valid, control = engine.requests
    assert list(valid.input_ids) == list(_StubVisionPrompt.input_ids)
    assert list(control.input_ids) != list(_StubVisionPrompt.input_ids)
    # The control carries its own image, so its rejection strands nothing.
    assert control.image_data != valid.image_data


@pytest.mark.asyncio
async def test_servers_are_probed_concurrently(monkeypatch):
    """Test that probe cost does not scale with the number of servers.

    Each server costs a full request round trip, so probing in sequence would
    add that wait per server to every training start.
    """
    _patch_probe_deps(monkeypatch)

    class _Slow(_InnerEngine):
        in_flight = 0
        peak = 0

        async def agenerate(self, request):
            type(self).in_flight += 1
            type(self).peak = max(type(self).peak, type(self).in_flight)
            try:
                await asyncio.sleep(0.05)
                return await super().agenerate(request)
            finally:
                type(self).in_flight -= 1

    engine = _Slow(["a:1", "b:2", "c:3"])
    await vision_canary.run_exact_token_canary(engine, _StubProcessor(), object())

    assert _Slow.peak == 3  # one in flight per address


@pytest.mark.asyncio
async def test_an_unrelated_failure_is_not_accepted_as_proof(monkeypatch):
    """Test that only the mismatch rejection certifies a server.

    A timeout, reset or 503 during the negative control would otherwise be read
    as "the server rejected it", quietly certifying a server that never checked.
    """
    _patch_probe_deps(monkeypatch)

    class _FlakyOnControl(_InnerEngine):
        async def agenerate(self, request):
            self.requests.append(request)
            if list(request.input_ids) != list(_StubVisionPrompt.input_ids):
                raise TimeoutError("connection reset")  # not a mismatch rejection
            return object()

    with pytest.raises(RuntimeError, match="not with the refusal that proves"):
        await vision_canary.run_exact_token_canary(
            _FlakyOnControl(["a:1"]), _StubProcessor(), object()
        )


@pytest.mark.asyncio
async def test_probe_covers_a_single_backend_engine(monkeypatch):
    """Test that an InfBridge-shaped engine is probed, not rejected.

    The data proxy passes one of these. Requiring ``addresses`` would make every
    multimodal lifespan fail before serving a single request.
    """
    _patch_probe_deps(monkeypatch)
    engine = _SingleBackendEngine()

    await vision_canary.run_exact_token_canary(engine, _StubProcessor(), object())

    # Probed exactly as a listed server would be: valid, then the control.
    assert len(engine.requests) == 2


@pytest.mark.asyncio
async def test_single_backend_engine_without_an_address_is_an_error(monkeypatch):
    """Test that an unconfigured bridge is reported, not silently passed."""
    _patch_probe_deps(monkeypatch)
    engine = _SingleBackendEngine(backend_addr="")

    with pytest.raises(RuntimeError, match="no server addresses"):
        await vision_canary.run_exact_token_canary(engine, _StubProcessor(), object())


@pytest.mark.asyncio
async def test_single_backend_engine_detects_a_missing_patch(monkeypatch):
    """Test that the negative control still works without rid pinning."""
    _patch_probe_deps(monkeypatch)

    class _Unpatched(_SingleBackendEngine):
        async def agenerate(self, request):
            self.requests.append(request)
            return object()

    with pytest.raises(RuntimeError, match="accepted a prompt"):
        await vision_canary.run_exact_token_canary(
            _Unpatched(), _StubProcessor(), object()
        )


@pytest.mark.asyncio
async def test_a_failing_probe_settles_its_siblings(monkeypatch):
    """Test that no probe is still running once the canary raises.

    The caller treats a raised canary as fatal and begins tearing the engine
    down, so a sibling left in flight would issue requests against an engine
    being disposed of.
    """
    _patch_probe_deps(monkeypatch)

    class _OneBadServer(_InnerEngine):
        live = 0

        async def agenerate(self, request):
            if "b:2" in self.rid_to_address.get(request.rid, ""):
                raise RuntimeError("that server is broken")
            type(self).live += 1
            try:
                await asyncio.sleep(5)  # outlives the failure unless cancelled
                return await super().agenerate(request)
            finally:
                type(self).live -= 1

    engine = _OneBadServer(["a:1", "b:2"])
    with pytest.raises(RuntimeError):
        await vision_canary.run_exact_token_canary(engine, _StubProcessor(), object())

    assert _OneBadServer.live == 0
    assert not engine.rid_to_address  # every pin cleaned up
