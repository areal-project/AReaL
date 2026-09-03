# SPDX-License-Identifier: Apache-2.0

"""Narrow lifecycle transaction for one MOPD teacher-scoring phase."""

from __future__ import annotations

from typing import Any, Protocol

from areal.api.cli_args import MOPDConfig
from areal.infra.rpc.rtensor import RTensorDrainReceipt
from areal.trainer.mopd.targets import MOPD_CONTRIBUTIONS_KEY
from areal.trainer.mopd.teacher_manager import (
    PersistentTeacherManager,
    TeacherController,
    TeacherManagerState,
)
from areal.utils import logging

logger = logging.getLogger("MOPDTeacherPhase")


class MOPDTargetActor(Protocol):
    """Actor operations required by the teacher transaction."""

    def assert_mopd_runtime_topology(self) -> None: ...

    def aggregate_mopd_targets(
        self, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]: ...

    def strict_clear_batches(self, *targets: Any) -> RTensorDrainReceipt: ...


class BatchDrainer(Protocol):
    """One role that may have localized an RTensor batch."""

    def strict_clear_batches(self, *targets: Any) -> RTensorDrainReceipt: ...


class MOPDTeacherPhase:
    """Select, score, aggregate, drain, and release MOPD teachers atomically."""

    def __init__(
        self,
        *,
        config: MOPDConfig,
        manager: PersistentTeacherManager,
        actor: MOPDTargetActor,
        critic: BatchDrainer | None = None,
        ref: BatchDrainer | None = None,
    ) -> None:
        self._config = config
        self._manager = manager
        self._actor = actor
        self._critic = critic
        self._ref = ref

    def materialize(self, rollout_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Materialize actor-owned MOPD targets and release teacher residency."""
        teacher_group_weights = self._resolve_teacher_group_weights(rollout_batch)
        required_teachers = [
            teacher_id
            for teacher_id in self._config.teachers
            if any(
                weights.get(teacher_id, 0.0) > 0 for weights in teacher_group_weights
            )
        ]
        if not required_teachers:
            raise ValueError("MOPD batch does not require any positive-weight teacher")

        teacher_outputs: list[Any] = []
        teacher_controllers: list[TeacherController] = []
        receipts: dict[str, RTensorDrainReceipt] = {}
        release_attempted = False

        def drain_critic() -> None:
            if self._critic is not None:
                receipts["critic"] = self._critic.strict_clear_batches(rollout_batch)

        def drain_ref() -> None:
            if self._ref is not None:
                receipts["ref"] = self._ref.strict_clear_batches(rollout_batch)

        def drain_teachers() -> None:
            for index, controller in enumerate(teacher_controllers):
                receipts[f"teacher:{index}"] = controller.strict_clear_batches(
                    rollout_batch, teacher_outputs
                )

        def drain_actor() -> None:
            receipts["actor"] = self._actor.strict_clear_batches(
                rollout_batch, teacher_outputs
            )

        try:
            self._actor.assert_mopd_runtime_topology()
            self._manager.pre_fetch(required_teachers[0])
            for teacher_index, teacher_id in enumerate(required_teachers):
                was_offloaded = self._manager.state is TeacherManagerState.OFFLOADED
                controller = self._manager.load(teacher_id)
                if all(existing is not controller for existing in teacher_controllers):
                    teacher_controllers.append(controller)
                if was_offloaded:
                    logger.info("[MOPD] teacher onload complete")
                controller.assert_mopd_runtime_topology()
                if teacher_index + 1 < len(required_teachers):
                    self._manager.pre_fetch(required_teachers[teacher_index + 1])

                indices = [
                    index
                    for index, weights in enumerate(teacher_group_weights)
                    if weights.get(teacher_id, 0.0) > 0
                ]
                subset = [
                    {
                        key: value
                        for key, value in rollout_batch[index].items()
                        if key != MOPD_CONTRIBUTIONS_KEY
                    }
                    for index in indices
                ]
                logps, dummy_logps = controller.compute_logp_padded(subset)
                if logps is None or len(logps) != len(indices):
                    raise RuntimeError(
                        f"MOPD teacher {teacher_id!r} returned an invalid logp batch"
                    )
                teacher_outputs.extend(logps)
                teacher_outputs.extend(dummy_logps)
                for index, logp in zip(indices, logps, strict=True):
                    rollout_batch[index].setdefault(MOPD_CONTRIBUTIONS_KEY, {})[
                        teacher_id
                    ] = {
                        "logp": logp,
                        "weight": teacher_group_weights[index][teacher_id],
                    }

            aggregated = self._actor.aggregate_mopd_targets(rollout_batch)
            drain_critic()
            drain_ref()
            drain_teachers()
            drain_actor()
            self._require_all_receipts(receipts, teacher_controllers)
            release_attempted = True
            self._manager.release(receipts["actor"])
            logger.info("[MOPD] teacher offload complete")
            return aggregated
        except BaseException:
            if self._critic is not None and "critic" not in receipts:
                self._emergency_drain("critic", drain_critic)
            if self._ref is not None and "ref" not in receipts:
                self._emergency_drain("reference", drain_ref)
            if any(
                f"teacher:{index}" not in receipts
                for index in range(len(teacher_controllers))
            ):
                self._emergency_drain("teacher", drain_teachers)
            if "actor" not in receipts:
                self._emergency_drain("actor", drain_actor)
            try:
                self._require_all_receipts(receipts, teacher_controllers)
            except RuntimeError:
                try:
                    self._manager.close()
                except Exception:
                    logger.error(
                        "MOPD teacher phase forced close failed", exc_info=True
                    )
            else:
                if release_attempted:
                    try:
                        self._manager.close()
                    except Exception:
                        logger.error(
                            "MOPD teacher phase release rollback failed",
                            exc_info=True,
                        )
                else:
                    try:
                        self._manager.release(receipts["actor"])
                    except Exception:
                        try:
                            self._manager.close()
                        except Exception:
                            logger.error(
                                "MOPD teacher phase release rollback failed",
                                exc_info=True,
                            )
            raise

    def close(self) -> None:
        """Close the persistent teacher manager owned by this phase."""
        self._manager.close()

    def _resolve_teacher_group_weights(
        self, rollout_batch: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        teacher_group_weights: list[dict[str, float]] = []
        for trajectory in rollout_batch:
            teacher_group = trajectory.get("mopd_route")
            if (
                not isinstance(teacher_group, str)
                or teacher_group not in self._config.teacher_groups
            ):
                raise ValueError(
                    f"Unknown or missing MOPD teacher group {teacher_group!r}"
                )
            teacher_group_weights.append(self._config.teacher_groups[teacher_group])
        return teacher_group_weights

    @staticmethod
    def _emergency_drain(
        role: str,
        drain: Any,
    ) -> None:
        try:
            drain()
        except Exception:
            logger.error(
                "MOPD emergency %s RTensor drain failed; forcing phase teardown",
                role,
                exc_info=True,
            )

    def _require_all_receipts(
        self,
        receipts: dict[str, RTensorDrainReceipt],
        teacher_controllers: list[TeacherController],
    ) -> None:
        expected = {"actor"}
        if self._critic is not None:
            expected.add("critic")
        if self._ref is not None:
            expected.add("ref")
        expected.update(f"teacher:{index}" for index in range(len(teacher_controllers)))
        missing = sorted(expected - receipts.keys())
        if missing:
            raise RuntimeError(
                "MOPD RTensor consumers did not acknowledge drain: "
                + ", ".join(missing)
            )
        expected_roles = {
            "actor": "actor",
            "critic": "critic",
            "ref": "ref",
            **{
                f"teacher:{index}": "mopd-teacher"
                for index in range(len(teacher_controllers))
            },
        }
        mismatched = [
            f"{key}={receipts[key].consumer_role!r}"
            for key, role in expected_roles.items()
            if key in expected and receipts[key].consumer_role != role
        ]
        if mismatched:
            raise RuntimeError(
                "MOPD RTensor drain receipts have unexpected consumer roles: "
                + ", ".join(mismatched)
            )
