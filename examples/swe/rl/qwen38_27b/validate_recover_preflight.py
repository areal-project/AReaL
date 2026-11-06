"""Read-only checks before a same-trial recovery; never submits or cancels jobs."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


class RecoveryPreflightError(RuntimeError):
    pass


JOB_FIELDS = ("controller_job_id", "actor_job_id", "rollout_job_id", "cleanup_job_id")
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "PREEMPTED",
    "REVOKED",
    "DEADLINE",
}


def previous_job_ids(submission: Path) -> list[str]:
    info = json.loads(submission.read_text())
    ids = [str(info.get(field, "")) for field in JOB_FIELDS]
    if any(not job_id.isdigit() for job_id in ids) or len(set(ids)) != len(ids):
        raise RecoveryPreflightError("Previous job IDs are missing or invalid")
    return ids


def previous_jobs_finished(submission: Path) -> dict:
    ids = previous_job_ids(submission)
    result = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "--format=JobID,State%32"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        cells = line.split("|")
        if len(cells) >= 2 and cells[0] in ids and cells[1].strip():
            states[cells[0]] = cells[1].split()[0]
    if set(states) != set(ids) or any(
        state not in TERMINAL_STATES for state in states.values()
    ):
        raise RecoveryPreflightError(
            "Previous training/cleanup jobs are not confirmed terminal"
        )
    return states


def previous_jobs_from_report(submission: Path, report_path: Path) -> dict:
    """Consume the immediately preceding host check without requiring Slurm in SIF."""
    ids = previous_job_ids(submission)
    report = json.loads(report_path.read_text())
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    if (
        report.get("scope") != "previous_jobs_only"
        or report.get("read_only") is not True
        or report.get("preflight_passed") is not True
        or report.get("submission_sha256") != digest
    ):
        raise RecoveryPreflightError(
            "Host job-state report does not match this recovery"
        )
    checked = datetime.fromisoformat(report["checked_at_utc"])
    age = (datetime.now(UTC) - checked).total_seconds()
    if not -5 <= age <= 300:
        raise RecoveryPreflightError(
            "Host job-state report is stale or from the future"
        )
    states = report.get("previous_jobs", {})
    if set(states) != set(ids) or any(
        state not in TERMINAL_STATES for state in states.values()
    ):
        raise RecoveryPreflightError(
            "Previous training/cleanup jobs are not confirmed terminal"
        )
    return states


def validate_manifest(checkpoint_root: Path, engine_names: list[str]) -> dict:
    from areal.utils import checkpoint_pointer
    from areal.utils.recover import RecoverInfo

    source = checkpoint_pointer.resolve_checkpoint(str(checkpoint_root), engine_names)
    if source is None:
        raise RecoveryPreflightError("This trial has no published checkpoint")
    if not source.transactional:
        raise RecoveryPreflightError(
            "This recovery requires the current trial's transactional checkpoint"
        )
    info = RecoverInfo.load(source.manifest)
    record = checkpoint_pointer.read_latest(str(checkpoint_root))
    if record is None or info.last_step_info.global_step != record.global_step:
        raise RecoveryPreflightError("Checkpoint pointer and manifest steps differ")
    step = info.last_step_info
    if (
        step.global_step < 0
        or step.epoch < 0
        or not 0 <= step.epoch_step < step.steps_per_epoch
    ):
        raise RecoveryPreflightError("Checkpoint step counters are invalid")
    if not info.dataloader_info:
        raise RecoveryPreflightError("Checkpoint dataloader state is empty")
    for payload in source.payloads.values():
        if not any(Path(payload).iterdir()):
            raise RecoveryPreflightError("Checkpoint engine payload directory is empty")
    return {
        "generation": record.generation,
        "saved_global_step": step.global_step,
        "next_global_step": step.next().global_step,
        "manifest_loaded": True,
        "engine_names": list(source.payloads),
        "payload_directories_nonempty": True,
        "full_optimizer_payload_and_resume_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--previous-jobs-only", action="store_true")
    parser.add_argument("--previous-jobs-report", type=Path)
    args = parser.parse_args()
    if args.previous_jobs_only and args.previous_jobs_report:
        parser.error("A host-only check cannot consume a previous host report")
    os.umask(0o077)
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "read_only": True,
        "scope": "previous_jobs_only" if args.previous_jobs_only else "full_preflight",
    }
    try:
        report["submission_sha256"] = hashlib.sha256(
            args.submission.read_bytes()
        ).hexdigest()
        report["previous_jobs"] = (
            previous_jobs_from_report(args.submission, args.previous_jobs_report)
            if args.previous_jobs_report
            else previous_jobs_finished(args.submission)
        )
        if not args.previous_jobs_only:
            if args.config is None:
                raise RecoveryPreflightError("A recovery config is required")
            from omegaconf import OmegaConf

            from examples.swe.utils import SWEPPOConfig

            from areal.api.cli_args import to_structured_cfg

            config = OmegaConf.to_object(
                to_structured_cfg(OmegaConf.load(args.config), SWEPPOConfig)
            )
            if config.recover.no_save_optim or config.recover.no_load_optim:
                raise RecoveryPreflightError(
                    "Full recovery requires optimizer save and load"
                )
            if not config.actor.megatron.use_checkpoint_opt_param_scheduler:
                raise RecoveryPreflightError(
                    "Full recovery requires LR scheduler restore"
                )
            checkpoint_root = (
                Path(config.recover.fileroot)
                / "checkpoints"
                / os.environ["USER"]
                / config.experiment_name
                / config.trial_name
            )
            names = ["default"]
            if getattr(config, "critic", None) is not None:
                names.append("critic")
            report["checkpoint"] = validate_manifest(checkpoint_root, names)
        report["preflight_passed"] = True
    except Exception as exc:
        report["preflight_passed"] = False
        report["error_type"] = type(exc).__name__
        # Only our fixed error strings are safe to print; backend errors can contain credentials.
        if isinstance(exc, RecoveryPreflightError):
            report["reason"] = str(exc)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["preflight_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
