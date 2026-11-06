"""Collect Pixel Reasoner image/video trajectories using AReaL's inference engine."""

import errno
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pixel_reasoner_workflow import PixelReasonerWorkflow
from transformers import AutoProcessor

from areal.api.cli_args import (
    ClusterSpecConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    RecoverConfig,
    SGLangConfig,
    StatsLoggerConfig,
    load_expr_config,
)
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import logging

logger = logging.getLogger("PixelReasoner")


@dataclass
class PixelReasonerConfig:
    """Inference-only configuration, without mandatory training datasets/savers."""

    experiment_name: str = "pixel-reasoner"
    trial_name: str = "trial1"
    seed: int = 1
    tokenizer_path: str = ""
    allocation_mode: str = "sglang:d4t2"
    cluster: ClusterSpecConfig = field(default_factory=ClusterSpecConfig)
    recover: RecoverConfig = field(default_factory=RecoverConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    stats_logger: StatsLoggerConfig = field(default_factory=StatsLoggerConfig)
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    data_root: str = ""
    manifest: str = "manifest.jsonl"
    dump_dir: str = "/tmp/areal/pixel-traces"
    max_turns: int | None = None
    max_output_tokens: int = 262144
    original_max_pixels: int = 1568000
    crop_max_pixels: int = 784000
    limit: int | None = None
    initial_concurrency: int | None = None
    concurrency_control: str | None = None
    controller_rank: int | None = None
    controller_world_size: int | None = None


def utc():
    return datetime.now(UTC).isoformat()


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except OSError as error:
            if error.errno != errno.ESTALE or attempt == 2:
                raise
            time.sleep(0.1)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def new_control(path, target, ceiling):
    root = Path(path).resolve().parent
    return dict(
        schema_version=1,
        run_id=root.name,
        run_dir=str(root),
        revision=uuid.uuid4().hex,
        target=target,
        ceiling=ceiling,
        requested_at_utc=utc(),
        requested_by_uid=os.getuid(),
        requested_by_pid=os.getpid(),
    )


def validate_control(value, path, ceiling, minimum=1):
    root = Path(path).resolve().parent
    if (
        value.get("schema_version") != 1
        or value.get("run_id") != root.name
        or value.get("run_dir") != str(root)
        or value.get("ceiling") != ceiling
        or not isinstance(value.get("revision"), str)
        or not value["revision"]
    ):
        raise ValueError("Concurrency control belongs to another run or ceiling")
    if (
        type(value.get("target")) is not int
        or not minimum <= value["target"] <= ceiling
    ):
        raise ValueError(f"Concurrency target must be in [{minimum}, {ceiling}]")
    return value


def quota(total, rank, workers):
    return total // workers + int(rank < total % workers)


class ConcurrencyControl:
    """Keep the last valid target; acknowledge revisions and publish live counters."""

    def __init__(self, path, initial, ceiling, rank, workers, logger):
        if type(initial) is not int or not workers <= initial <= ceiling:
            raise ValueError(
                "Initial concurrency must fit active controllers and ceiling"
            )
        self.path = Path(path).resolve() if path else None
        self.target, self.ceiling = initial, ceiling
        self.rank, self.workers, self.logger = rank, workers, logger
        self.revision, self.error, self.last_poll = None, None, -float("inf")
        self.applied_at, self.last_status, self.last_write = None, None, -float("inf")
        self.pending_history = False
        self.poll()

    def poll(self):
        if not self.path or time.monotonic() - self.last_poll < 1:
            return quota(self.target, self.rank, self.workers)
        self.last_poll = time.monotonic()
        try:
            value = validate_control(
                read_json(self.path), self.path, self.ceiling, self.workers
            )
            if value["revision"] == self.revision and value["target"] != self.target:
                raise ValueError("Changing target requires a new revision")
            if value["revision"] != self.revision:
                self.target, self.revision = value["target"], value["revision"]
                self.applied_at, self.pending_history = utc(), True
                self.logger.info(
                    "PIXEL_CONCURRENCY_APPLIED rank=%d total=%d revision=%s",
                    self.rank,
                    self.target,
                    self.revision,
                )
            self.error = None
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            message = f"{type(error).__name__}: {error}"
            if message != self.error:
                self.logger.warning(
                    "PIXEL_CONTROL_ERROR keeping_target=%d %s", self.target, message
                )
            self.error = message
        return quota(self.target, self.rank, self.workers)

    def snapshot(self):
        return dict(
            revision=self.revision,
            target=self.target,
            local_target=quota(self.target, self.rank, self.workers),
            applied_at_utc=self.applied_at,
        )

    def publish(self, in_flight, submitted, finished, phase="collecting"):
        if not self.path:
            return
        state = dict(
            self.snapshot(),
            rank=self.rank,
            workers=self.workers,
            ceiling=self.ceiling,
            in_flight=in_flight,
            submitted=submitted,
            finished=finished,
            phase=phase,
            control_error=self.error,
        )
        if state == self.last_status and time.monotonic() - self.last_write < 1:
            return
        directory = self.path.parent / "concurrency-status"
        directory.mkdir(exist_ok=True)
        state.update(run_id=self.path.parent.name, updated_at=time.time())
        write_json(directory / f"worker-{self.rank}.json", state)
        if self.pending_history:
            append_json(directory / f"worker-{self.rank}.jsonl", state)
            self.pending_history = False
        self.last_status = {
            key: value
            for key, value in state.items()
            if key not in ("run_id", "updated_at")
        }
        self.last_write = time.monotonic()


def collect(engine, workflow, rows, control):
    """Refill on any completion; lowering the target only pauses new admissions."""
    rows = iter(rows)
    in_flight = submitted = finished = 0
    exhausted, phase = False, "failed"
    try:
        while True:
            target = control.poll()
            while in_flight < target and not exhausted:
                row = next(rows, None)
                if row is None:
                    exhausted = True
                    break
                engine.submit(
                    dict(row, concurrency_at_dispatch=control.snapshot()), workflow
                )
                submitted += 1
                in_flight += 1
            control.publish(in_flight, submitted, finished)
            if exhausted and not in_flight:
                phase = "finished"
                break
            # An empty list is a timeout; [None] is one completed trace-only episode.
            results = engine.wait(1, timeout=0.5, raise_timeout=False)
            in_flight -= len(results)
            finished += len(results)
        return finished
    finally:
        control.publish(in_flight, submitted, finished, phase)


def input_rows(config, rank, workers):
    eligible_count = 0
    with (Path(config.data_root) / config.manifest).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("image"):
                continue
            if config.limit is not None and eligible_count >= config.limit:
                break
            owner = eligible_count % workers
            eligible_count += 1
            if owner == rank:
                yield row


def main(args: list[str]):
    config, _ = load_expr_config(args, PixelReasonerConfig)
    rank = (
        config.controller_rank
        if config.controller_rank is not None
        else int(os.environ.get("RANK", "0"))
    )
    world_size = (
        config.controller_world_size
        if config.controller_world_size is not None
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if not 0 <= rank < world_size:
        raise ValueError("Invalid controller rank/world size")
    concurrency = config.rollout.max_concurrent_rollouts or 1
    if concurrency < 1:
        raise ValueError("max_concurrent_rollouts must be positive")
    initial = (
        config.initial_concurrency
        if config.initial_concurrency is not None
        else concurrency
    )
    workers = min(world_size, initial)
    if not 1 <= initial <= concurrency:
        raise ValueError(
            "initial_concurrency must be between 1 and the rollout ceiling"
        )
    if rank >= workers:
        return
    # Slurm eval can launch multiple CPU controllers. Split both inputs and
    # the global concurrency budget so every task runs exactly once.
    config.rollout.max_concurrent_rollouts = quota(concurrency, rank, workers)
    control = ConcurrencyControl(
        config.concurrency_control, initial, concurrency, rank, workers, logger
    )
    processor = AutoProcessor.from_pretrained(
        config.tokenizer_path, trust_remote_code=True
    )
    workflow = PixelReasonerWorkflow(
        processor,
        config.gconfig,
        config.data_root,
        config.dump_dir,
        config.max_turns,
        config.max_output_tokens,
        config.original_max_pixels,
        config.crop_max_pixels,
    )
    engine = RemoteSGLangEngine(config.rollout)
    addresses = os.environ.get("AREAL_LLM_SERVER_ADDRS")
    if addresses:
        addresses = addresses.split(",")
    else:
        from areal.api.alloc_mode import ModelAllocation
        from areal.infra.utils.launcher import wait_llm_server_addrs

        replicas = ModelAllocation.from_str(
            config.rollout.backend, name="rollout"
        ).parallel.data_parallel_size
        addresses = wait_llm_server_addrs(
            config.experiment_name,
            config.trial_name,
            n_rollout_servers=replicas,
            timeout=config.rollout.setup_timeout,
        )
    engine.initialize(addr=addresses, train_data_parallel_size=1)
    try:
        count = collect(engine, workflow, input_rows(config, rank, workers), control)
        logger.info("Finished %d attempts; traces: %s", count, config.dump_dir)
    finally:
        engine.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
