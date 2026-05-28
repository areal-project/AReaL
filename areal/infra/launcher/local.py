# SPDX-License-Identifier: Apache-2.0

import getpass
import os
import re
import signal as signal_module
import subprocess
import sys
import time
import warnings
from collections import defaultdict

import psutil

from areal.api.alloc_mode import AllocationType, _AllocationMode
from areal.api.cli_args import (
    ClusterSpecConfig,
    InferenceEngineConfig,
    RecoverConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
    vLLMConfig,
)
from areal.infra.platforms import current_platform
from areal.infra.utils.exp_metadata import save_experiment_metadata
from areal.infra.utils.launcher import (
    BASE_ENVIRONS,
    JobException,
    JobInfo,
    JobState,
    get_scheduling_spec,
    get_thread_env_vars,
    validate_config_for_launcher,
    wait_llm_server_addrs,
)
from areal.utils import logging, name_resolve, names
from areal.utils.network import find_free_ports
from areal.utils.offload import get_tms_env_vars
from areal.utils.recover import check_if_recover

logger = logging.getLogger("LocalLauncher")
JOB_STATE_TO_PROCESS_STATUS = {
    JobState.NOT_FOUND: [],
    JobState.PENDING: [psutil.STATUS_PARKED],
    JobState.RUNNING: [
        psutil.STATUS_RUNNING,
        psutil.STATUS_SLEEPING,
        psutil.STATUS_DISK_SLEEP,
        psutil.STATUS_TRACING_STOP,
        psutil.STATUS_WAKING,
        psutil.STATUS_WAITING,
        psutil.STATUS_LOCKED,
        psutil.STATUS_IDLE,
    ],
    JobState.COMPLETED: [
        psutil.STATUS_DEAD,
        psutil.STATUS_STOPPED,
        psutil.STATUS_ZOMBIE,
    ],
    JobState.FAILED: [],
    JobState.CANCELLED: [],
}
RECOVER_TIME_INTERVAL = 10  # seconds

PROCESS_STATUS_TO_JOB_STATE = {}
for job_state, process_statuses in JOB_STATE_TO_PROCESS_STATUS.items():
    for process_status in process_statuses:
        PROCESS_STATUS_TO_JOB_STATE[process_status] = job_state


def terminate_process_and_children(pid: int, signal: str | int | None = None):
    if signal is None:
        signal = signal_module.SIGKILL
    if isinstance(signal, str):
        signal = getattr(signal_module, signal)
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            terminate_process_and_children(child.pid, signal)
        parent.send_signal(signal)
    except psutil.NoSuchProcess:
        pass


class LocalLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        self._jobs: dict[str, subprocess.Popen] = {}
        self._job_counter: dict[str, int] = defaultdict(int)
        self._job_states = {}

        self._gpu_counter = 0
        self._gpu_devices: list[str] = os.environ.get(
            current_platform.device_control_env_var,
            ",".join(map(str, range(current_platform.device_count()))),
        ).split(",")
        if len(self._gpu_devices) < 1:
            raise RuntimeError(
                f"Local mode can only run when there is at least one GPU. "
                f"{current_platform.device_control_env_var} is currently"
                f" set to: `{os.environ.get(current_platform.device_control_env_var, '')}`."
            )

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")

    def __del__(self):
        self.wait()

    def submit_array(
        self,
        job_name: str,
        cmd: str | list[str],
        count: int = 1,
        gpu: int = 0,
        env_vars: dict | None = None,
    ):
        if env_vars is None:
            env_vars = {}
        if not isinstance(cmd, list):
            cmd = [cmd] * count
        offset = self._job_counter[job_name]
        for i in range(count):
            if gpu > 0:
                # Allocate GPUs in a round-robin manner
                visible_devices = []
                for _ in range(gpu):
                    available_device_id = self._gpu_counter % len(self._gpu_devices)
                    self._gpu_counter += 1
                    visible_devices.append(available_device_id)
                env_vars[current_platform.device_control_env_var] = ",".join(
                    str(self._gpu_devices[j]) for j in visible_devices
                )
            c = (
                " ".join(str(k) + "=" + str(v) for k, v in env_vars.items())
                + " stdbuf -oL "
                + cmd[i]
            )
            c = f"{c} 2>&1 | tee -a {self.log_path_of(job_name)}"
            logger.info("Starting local process with command: %s", c)
            process = subprocess.Popen(
                c, shell=isinstance(c, str), stdout=sys.stdout, stderr=sys.stdout
            )
            self._jobs[f"{job_name}/{offset + i}"] = process
            self._job_counter[job_name] += 1

    def submit(
        self,
        job_name: str,
        cmd: str | list[str],
        gpu: int = 0,
        env_vars: dict | None = None,
    ):
        self.submit_array(job_name=job_name, cmd=cmd, gpu=gpu, env_vars=env_vars)

    def stop(self, job_name, signal=None):
        assert any(k.startswith(job_name) for k in self._jobs)
        keys = [k for k, p in self._jobs.items() if k.startswith(job_name)]
        procs = [p for k, p in self._jobs.items() if k.startswith(job_name)]
        logger.info(
            f"Stopping local process with signal {signal if signal else 'SIGKILL'}, "
            f"pid: {[p.pid for p in procs]}"
        )
        for p in procs:
            terminate_process_and_children(p.pid, signal=signal)
        for p in procs:
            p.wait()
        for k, p in zip(keys, procs):
            self._jobs.pop(k)
            del p

    def stop_all(self, signal=None):
        # signal argument is ignored in local stop_all
        for name in self._job_counter:
            self.stop(name, signal=signal)

    def find(self, job_name):
        if job_name in self._jobs:
            return JobInfo(name=job_name, state=JobState.RUNNING, host="localhost")
        else:
            return JobInfo(name=job_name, state=JobState.NOT_FOUND)

    def find_all(self, job_name_regex=".*"):
        rs = []
        for name in self._jobs:
            if re.fullmatch(job_name_regex, name):
                rs.append(self.find(name))
        return rs

    def wait(
        self,
        timeout=None,
        check_status: tuple[JobState, ...] = (
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ),
        remove_status: tuple[JobState, ...] = (JobState.COMPLETED,),
        update=False,
    ):
        deadline = None if timeout is None else time.time() + timeout
        logger.info(
            "Waiting for %d local running processes, pids: %s",
            len(self._jobs),
            " ".join(str(job.pid) for job in self._jobs.values()),
        )
        left = set(self._jobs.keys())
        num_jobs_left = len(left)

        while len(left) > 0:
            to_remove = []
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.run_name}: {', '.join(sorted(left))}"
                )
            # update job states
            for job_name in list(left):
                job = self._jobs[job_name]
                pid = job.pid
                process = psutil.Process(pid)
                self._job_states[job_name] = PROCESS_STATUS_TO_JOB_STATE.get(
                    process.status(), JobState.NOT_FOUND
                )

            for job_name in list(left):
                state = self._job_states[job_name]
                if state in check_status:
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=job_name.split("/")[0],
                        host="local",
                        reason=state,
                    )
                if state in remove_status:
                    logger.info(f"Job {job_name} is {state}.(Removed)")
                    left.remove(job_name)
                    to_remove.append(job_name)

            if update:
                for k in to_remove:
                    self._jobs.pop(k)
                    worker_type = k.split("/")[0]
                    assert worker_type in self._job_counter
                    self._job_counter[worker_type] -= 1
                    if self._job_counter[worker_type] <= 0:
                        self._job_counter.pop(worker_type)

            time.sleep(2)


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    local_main(config, run_id=0)


def local_main(config, run_id: int = 0):
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    is_recover_run = check_if_recover(config.recover, run_id)
    warnings.warn(
        "SPMD launchers use the deprecated _AllocationMode parser which will be removed. "
        "Bare dimension strings (e.g., 'd4t2') are NO LONGER ACCEPTED. "
        "All allocation strings must include an explicit backend prefix "
        "(e.g., 'fsdp:d4', 'sglang:d4t2'). "
        "Migrate to single-controller mode (scheduler.type=local) with per-engine 'backend' "
        "fields (e.g., actor.backend='fsdp:d4'). See docs/en/reference/alloc_mode.md.",
        FutureWarning,
        stacklevel=2,
    )
    validate_config_for_launcher(config)
    launcher = LocalLauncher(
        config.experiment_name, config.trial_name, config.cluster.fileroot
    )

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )
    alloc_mode = _AllocationMode.from_str(config.allocation_mode)

    logger.info(
        f"LocalLauncher: experiment_name={config.experiment_name}, "
        f"trial_name={config.trial_name}, fileroot={config.cluster.fileroot}, "
        f"run_id={run_id}, is_recover_run={is_recover_run}"
    )

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    server_addrs = []
    if alloc_mode.gen_backend in ("sglang", "vllm"):
        # Launcher should launch llm servers according to allocation mode.
        if alloc_mode.gen_backend == "sglang":
            config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
            random_seed = config.sglang.random_seed
        else:
            config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
            random_seed = config.vllm.seed

        config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)

        backend_spec = {
            "sglang": {
                "module": "areal.infra.launcher.sglang_server",
                "seed_arg": "sglang.random_seed",
                "set_device_env": False,
            },
            "vllm": {
                "module": "areal.infra.launcher.vllm_server",
                "seed_arg": "vllm.seed",
                "set_device_env": True,  # vLLM needs `device_control_env_var` to control GPU allocation
            },
        }

        spec = backend_spec[alloc_mode.gen_backend]

        base_seed = random_seed
        seed_arg = spec["seed_arg"]
        module = spec["module"]
        server_cmd = (
            f"python3 -m {module} {' '.join(sys.argv[1:])} {seed_arg}={base_seed}"
        )

        # Launch inference servers.
        rollout_spec = get_scheduling_spec(config.rollout)
        rollout_env_vars = rollout_spec.env_vars
        rollout_cpus_per_task = rollout_spec.cpu
        thread_env = get_thread_env_vars(
            cpus_per_task=rollout_cpus_per_task,
            existing_env_vars=rollout_env_vars,
        )
        launcher.submit_array(
            job_name="llm_server",
            cmd=server_cmd,
            count=1,
            gpu=alloc_mode.gen.pp_size
            * alloc_mode.gen.tp_size
            * alloc_mode.gen.dp_size,
            env_vars={**BASE_ENVIRONS, **thread_env, **rollout_env_vars},
        )

        # Get llm server addresses by name resolve
        try:
            server_addrs = wait_llm_server_addrs(
                config.experiment_name,
                config.trial_name,
                n_rollout_servers=alloc_mode.gen.dp_size,
            )
            logger.info(
                f"LLM inference server launched at: AREAL_LLM_SERVER_ADDRS={','.join(server_addrs)}"
            )
        except (TimeoutError, KeyboardInterrupt) as e:
            launcher.stop_all(signal="SIGINT")
            raise e

    if config.get("enable_offload", False):
        tms_env_vars = get_tms_env_vars()
    else:
        tms_env_vars = {}
    # Launch trainer entrypoint
    if alloc_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        gpu = nprocs = alloc_mode.train.world_size
        _env_vars = dict(
            AREAL_LLM_SERVER_ADDRS=",".join(server_addrs),
        )
        if alloc_mode.gen_backend == "sglang":
            # Required by NCCL weight update group.
            _env_vars["NCCL_CUMEM_ENABLE"] = "0"
            _env_vars["NCCL_NVLS_ENABLE"] = "0"
        # All experiment configs should have the `actor` field.
        actor_spec = get_scheduling_spec(config.actor)
        actor_env_vars = actor_spec.env_vars
        actor_cpus_per_task = actor_spec.cpu
        thread_env = get_thread_env_vars(
            cpus_per_task=actor_cpus_per_task,
            existing_env_vars=actor_env_vars,
        )
        launcher.submit(
            job_name="trainer",
            cmd=f"torchrun --nnodes 1 --nproc-per-node {nprocs} --master-addr localhost --master-port {find_free_ports(1, (10000, 32767))[0]} {' '.join(sys.argv[1:])}",
            gpu=gpu,
            env_vars={
                **BASE_ENVIRONS,
                **thread_env,
                **actor_env_vars,
                **_env_vars,
                **tms_env_vars,
                "AREAL_SPMD_MODE": "1",
            },
        )

    try:
        launcher.wait(
            check_status=(
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.NOT_FOUND,
                JobState.COMPLETED,
            ),
            remove_status=(),
        )
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        launcher.stop_all("SIGTERM")
        # NOTE: For local launcher, We cannot distinguish between a completed job and a failed job.
        # So we will always try to recover the job if it is finished or failed.
        recover_states = [JobState.FAILED, JobState.NOT_FOUND, JobState.COMPLETED]
        if isinstance(e, JobException):
            recover_this = (
                e.reason in recover_states
                and run_id < config.recover.retries
                and config.recover.mode in ("on", "auto")
            )
            if recover_this:
                time.sleep(RECOVER_TIME_INTERVAL)
                local_main(config, run_id=run_id + 1)
            else:
                raise e
        else:
            raise e


if __name__ == "__main__":
    main()
