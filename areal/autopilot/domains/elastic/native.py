# SPDX-License-Identifier: Apache-2.0

"""Accepted Online Reference activation at native published-update boundaries."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from areal.autopilot.domains.elastic.bundle import ReconfiguratorApplication
from areal.autopilot.domains.elastic.contracts import deployment_contract
from areal.autopilot.revision.contracts import digest
from areal.elastic.contracts import ControlBusy
from areal.elastic.infra.activation import (
    ReconfiguratorBoundaryBusy,
    ReconfiguratorRequest,
)
from areal.elastic.infra.adapter import NativeControlEnvironment, ReconfiguratorAdapter


class _RevisionHooks:
    def __init__(self, adapter):
        self.adapter = adapter

    def poll(self):
        a = self.adapter
        if a.revision is not None and a.measurement_ready:
            a.revision.tick(a)

    def recover(self):
        self.adapter.recover()


class RevisionAdapter(ReconfiguratorAdapter):
    """Add accepted component activation to the ordinary Online Runtime adapter."""

    supports_reconfigurator_activation = True

    def __init__(self, *, revision=None, bundle=None, **kwargs):
        if bundle is not None:
            import json

            from areal.autopilot.domains.elastic.bundle import load_bundle
            from areal.autopilot.domains.elastic.contracts import decode
            from areal.elastic.contracts import OnlineLayout

            value = json.loads(Path(bundle).read_text())
            loaded = load_bundle(bundle)
            spec = value["assembly"]
            defaults = dict(
                implementation="reference",
                models=loaded.models,
                run_config=loaded.config,
                handoff=loaded.handoff,
                slo=loaded.slo,
                settings=spec.get("reconfigurator", {}),
                candidates=tuple(decode(OnlineLayout, c) for c in spec["candidates"]),
            )
            if "replay" in spec:
                defaults["replay"] = spec["replay"]
            if "forecast" in spec:
                defaults["forecast"] = spec["forecast"]
            kwargs = {**defaults, **{k: v for k, v in kwargs.items() if v is not None}}
        super().__init__(**kwargs)
        self.revision, self.bundle = revision, bundle
        self.hooks = _RevisionHooks(self)
        self.reconfigurator_application = None
        self.failed = False

    def assembly(self):
        if self.run_config is None or self.handoff is None:
            raise ValueError(
                "Online revision requires explicit run and transition inputs"
            )
        r = self.reconfigurator
        from areal.elastic.reconfigurator.reference.forecast import HistoricalReplay

        key = "replay" if isinstance(r.forecaster, HistoricalReplay) else "forecast"
        return dict(
            run=dataclasses.asdict(self.run_config),
            handoff=dataclasses.asdict(
                self.placements.prediction_handoff(self.handoff)
            ),
            slo=dataclasses.asdict(r.slo),
            candidates=[dataclasses.asdict(c) for c in r.candidates],
            reconfigurator=dataclasses.asdict(r.config),
            **{key: dataclasses.asdict(r.forecaster.config)},
            deployment=dict(
                compatibility=self.owner.compatibility,
                native_execution_id=self.execution_id,
                native_runtime_id=self.owner.identity,
                gpu_reservation=list(self.owner.resource_reservation),
            ),
        )

    def bind(self, owner):
        super().bind(owner)
        if self.implementation != "reference":
            raise ValueError(
                "Revision currently targets reference; Basic v1 revision integration is deferred"
            )
        contract = (
            None
            if self.run_config is None or self.handoff is None
            else digest(deployment_contract(self.assembly()))
        )
        self.reconfigurator_application = ReconfiguratorApplication(
            owner.root / "reconfigurator-application",
            compatibility=owner.compatibility,
            execution_context=dict(
                scope="online-reference-v1",
                native_runtime_id=owner.identity,
                queue_id=owner.inbox.queue.queue_id,
                checkpoint_root=str(owner.store.root),
                online_contract_sha256=contract,
            ),
            initial_bundle_id=digest(
                dict(
                    implementation="reference",
                    online_contract_sha256=contract,
                    native_runtime_id=owner.identity,
                )
            ),
        )
        self.recover()
        if (
            self.bundle is not None
            and self.reconfigurator_application.state()["bundle_id"]
            != self.bundle_id_from_path()
        ):
            prepared = self.prepare_reconfigurator(Path(self.bundle))
            self.apply_reconfigurator(
                ReconfiguratorRequest(
                    "startup-" + prepared.bundle_id[:16],
                    prepared.bundle_id,
                    self.reconfigurator_generation,
                    owner.applied_version,
                ),
                prepared,
            )

    def bundle_id_from_path(self):
        import json

        return json.loads(Path(self.bundle).read_text())["bundle_id"]

    @property
    def runtime_history(self):
        return self.owner.monitor.runtime_history

    @property
    def measurement_ready(self):
        return self.owner.monitor.metrics_available

    @property
    def weights_version(self):
        return self.owner.applied_version

    @property
    def reconfigurator_generation(self):
        return (
            0
            if self.reconfigurator_application is None
            else self.reconfigurator_application.state()["generation"]
        )

    @property
    def reconfigurator_boundary_ready(self):
        return (
            not self.failed
            and not self.closed
            and self.measurement_ready
            and self.owner.control_snapshot().boundary_ready
            and (self.controller is None or self.controller.pending is None)
        )

    def measurement_failed(self, error):
        self.owner.monitor.mark_unavailable(error)

    def reconfigurator_metadata(self):
        return {
            **super().reconfigurator_metadata(),
            "reconfigurator_generation": self.reconfigurator_generation,
        }

    def observe(self, batch=None, *, now=None):
        view = super().observe(batch, now=now)
        return dataclasses.replace(view, busy=view.busy or self.failed)

    def _require_decision_boundary(self):
        super()._require_decision_boundary()
        if self.failed:
            raise RuntimeError(
                "Recover the durable reconfigurator activation before deciding"
            )

    def prepare_reconfigurator(self, bundle):
        prepared = self.reconfigurator_application.prepare(Path(bundle))
        loaded = prepared.reconfigurator
        if (
            loaded.config.batch_size != self.owner.inbox.batch_size
            or loaded.config.staleness != self.owner.inbox.max_staleness
        ):
            raise ValueError("Online bundle changes native training semantics")
        if any(
            self.placements.resolve(c) is None for c in loaded.reconfigurator.candidates
        ):
            raise ValueError("Online bundle requires an unsupported native layout")
        # Shadow decision uses a private copy: probing cannot advance live history.
        import copy

        shadow = copy.deepcopy(loaded.reconfigurator)
        shadow.decide(
            self.observe(),
            shadow.make_model(loaded.models, loaded.config, loaded.handoff, loaded.slo),
        )
        return dataclasses.replace(prepared, probed=True)

    def _install(self, loaded):
        old = self.reconfigurator
        new = loaded.reconfigurator
        # Component revisions retain the live observations, role-history clocks and search index.
        for name in (
            "forecaster",
            "decision_index",
            "last_switch",
            "last_role_switch",
            "last_search",
            "_winners",
        ):
            setattr(new, name, getattr(old, name))
        new.search_invalidated()
        if self.controller is not None:
            self.controller.cancel()
        self.reconfigurator, self.models = new, loaded.models
        self.run_config, self.handoff, self.slo = (
            loaded.config,
            loaded.handoff,
            loaded.slo,
        )
        self.bundle_id = loaded.bundle_id
        self.environment = NativeControlEnvironment(self)
        self.controller = new.bind(self.environment)

    def apply_reconfigurator(self, request, prepared):
        try:
            with self.owner.control_operation():
                previous = self.reconfigurator_application.preflight(
                    request, self.weights_version
                )
                if previous is not None:
                    return previous
                if not self.reconfigurator_boundary_ready:
                    raise ReconfiguratorBoundaryBusy(
                        "Activation waits for a published update with no pending Online decision"
                    )
                try:
                    receipt = self.reconfigurator_application.commit(
                        request, self.weights_version, prepared, self.history
                    )
                    self._install(prepared.reconfigurator)
                    return receipt
                except BaseException:
                    self.failed = True
                    self.owner.fence_admission()
                    raise
        except ControlBusy as error:
            raise ReconfiguratorBoundaryBusy(str(error)) from error

    def recover(self):
        if self.reconfigurator_application is None:
            return
        prepared = self.reconfigurator_application.recover()
        if prepared is not None:
            self._install(prepared.reconfigurator)
        self.failed = False

    def close(self):
        if self.closed:
            return
        try:
            if self.revision is not None:
                self.revision.close()
        finally:
            super().close()
