# SPDX-License-Identifier: Apache-2.0

import copy
import getpass
import os
import re
import subprocess
import sys
import time
import warnings

import areal.utils.logging as logging
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
    validate_config_for_distributed_launcher,
    wait_llm_server_addrs,
)
from areal.infra.utils.slurm import (
    APPTAINER_CMD_TEMPLATE,
    SBATCH_SCRIPT_TEMPLATE,
    SRUN_CMD_TEMPLATE,
    cancel_jobs,
    query_jobs,
)
from areal.utils import name_resolve, names
from areal.utils.offload import get_tms_env_vars
from areal.utils.recover import check_if_recover

logger = logging.getLogger("SlurmLauncher")

SLURM_WAIT_CHECK_TIME_INTERVAL = 5  # seconds
RECOVER_TIME_INTERVAL = 10  # seconds


class SlurmLauncher:
    def __init__(
        self, experiment_name: str, trial_name: str, fileroot: str, container_type: str
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.container_type = container_type

        # slurm_job_id -> JobInfo
        self.jobs: dict[int, JobInfo] = {}
        self.job_names = []

    @property
    def run_name(self) -> str:
        """Returns the run name of this launcher."""
        return f"{self.experiment_name}_{self.trial_name}"

    def slurm_name(self, job_name: str) -> str:
        """Returns the slurm name of a job."""
        return f"{self.experiment_name}_{self.trial_name}:{job_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")

    def sbatch_path_of(self, job_name: str) -> str:
        sbatch_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(sbatch_path, exist_ok=True)
        return os.path.join(sbatch_path, f"{job_name}.sh")

    def submit(self, job_name, cmd, **kwargs):
        """Submits and launch a job with SBATCH.

        Args:
            cmd (str or List[str]): The core command to be executed.
        """
        return self.submit_array(job_name, cmd, count=1, **kwargs)

    def find_job_id(self, job_name: str):
        job_name = self.slurm_name(job_name)
        for job_id, job_info in self.jobs.items():
            if job_info.name == job_name:
                return job_id
        return None

    def submit_array(
        self,
        job_name: str,
        cmd: list[str] | str,
        count: int,
        nodes: int,
        n_gpus_per_node: int,
        cpus_per_task: int,
        mem_per_task: int,  # MB
        container_image: str,
        srun_additional_args: str = "",
        container_mounts: str | None = None,
        env_vars: dict | list[dict] | None = None,
        nodelist: str | None = None,
        exclude: str | None = None,
    ):
        """Submits and launch a job array with SBATCH.
        Note that a job array has one (unique) slurm name, and one (unique) slurm id.

        Args:
            job_name (str): The job name of the job array. The actual slurm name will be
                `<experiment_name>_<trial_name>:<job_name>`.
            cmd (str or List[str]): The core command to be executed.
            count (int): The number of jobs in the array.
        """
        assert job_name not in self.job_names, (
            f"Job {job_name} is already submitted. "
            "Please use a different job name or stop the existing job."
        )
        if isinstance(cmd, str):
            cmd = [cmd]
        assert len(cmd) == count, (
            f"Command length {len(cmd)} does not match the job count {count}. "
            "Please provide a command for each job in the array."
        )
        assert count % nodes == 0, (
            f"Job count {count} must be divisible by the number of nodes {nodes}. "
            "Please adjust the job count or the number of nodes."
        )
        ntasks_per_node = count // nodes
        assert n_gpus_per_node % ntasks_per_node == 0, (
            "GPUs must be evenly distributed across tasks. "
            f"Current #GPUs per node {n_gpus_per_node}, #tasks per node {ntasks_per_node}."
        )

        mem_per_cpu = mem_per_task // cpus_per_task  # MB per CPU
        mem_per_node = (
            mem_per_task * count // nodes + 1024 * 10
        )  # make sure slurm does not run out of resources

        sbatch_options = [
            f"--job-name={self.slurm_name(job_name)}",
            f"--output={self.log_path_of(job_name)}",
            "--open-mode=append",
            "--no-requeue",
            f"--nodes={nodes}-{nodes}",
            f"--ntasks-per-node={ntasks_per_node}",
            f"--gres=gpu:{n_gpus_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem_per_node}M",
        ]

        if nodelist:
            sbatch_options.append(f"--nodelist={nodelist}")
        if exclude:
            sbatch_options.append(f"--exclude={exclude}")

        sbatch_options_str = "\n".join([f"#SBATCH {opt}" for opt in sbatch_options])

        if env_vars is None:
            env_vars = dict()
        n_gpus_per_task = n_gpus_per_node // ntasks_per_node
        if current_platform.device_control_env_var in env_vars:
            raise ValueError(
                f"{current_platform.device_control_env_var} should be automatically resolved by Launcher instead of manually assigned."
            )

        srun_cmds = []
        for i in range(count):
            node_id = i // ntasks_per_node
            # Prepare the command for each job in the array
            job_cmd = cmd[i]

            if isinstance(env_vars, list):
                _env_vars = env_vars[i]
            else:
                _env_vars = env_vars

            if self.container_type == "apptainer":
                env_string = " ".join(f"--env {k}={v}" for k, v in _env_vars.items())
                apptainer_cmd = APPTAINER_CMD_TEMPLATE.format(
                    container_mounts=container_mounts or "",
                    container_env_strings=env_string,
                    container_image=container_image,
                    cmd=job_cmd,
                )
                srun_cmd = SRUN_CMD_TEMPLATE.format(
                    additional_args=srun_additional_args,
                    nodes=1,
                    ntasks=1,
                    node_id=node_id,
                    n_gpus_per_node=n_gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    mem_per_cpu=mem_per_cpu,
                    cmd=apptainer_cmd,
                )
            elif self.container_type == "none":
                env_string = "--export=" + ",".join(
                    f"{k}={v}" for k, v in _env_vars.items()
                )
                srun_additional_args = srun_additional_args + " " + env_string
                srun_cmd = SRUN_CMD_TEMPLATE.format(
                    additional_args=srun_additional_args,
                    nodes=1,
                    ntasks=1,
                    node_id=node_id,
                    n_gpus_per_node=n_gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    mem_per_cpu=mem_per_cpu,
                    cmd=job_cmd,
                )
            else:
                raise ValueError(
                    f"Unsupported container type: {self.container_type}. "
                    "Supported types are 'apptainer' and 'none'."
                )
            srun_cmds.append(srun_cmd)

        srun_cmds = "\n".join(srun_cmds)
        sbatch_script = SBATCH_SCRIPT_TEMPLATE.format(
            sbatch_options=sbatch_options_str,
            srun_additional_args=srun_additional_args,
            srun_cmds=srun_cmds,
        )
        sbatch_file_path = self.sbatch_path_of(f"{job_name}")
        with open(sbatch_file_path, "w") as f:
            f.write(sbatch_script)

        # Submit the job
        try:
            output = (
                subprocess.check_output(["sbatch", sbatch_file_path])
                .decode("utf-8")
                .strip()
            )
            logger.info(
                f"Submitted Slurm job {self.slurm_name(job_name)} to scheduler. To check the output, run \n\t`tail -f {self.log_path_of(job_name)}`."
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Failed to submit job {self.slurm_name(job_name)}. "
                f"For debugging, please make sure your sbatch command works "
                f"and check generated sbatch file on {sbatch_file_path}."
            )
            logger.error(f"Error message: {e}")
            return

        match = re.search(r"Submitted batch job (\d+)", output)
        slurm_job_id = int(match.group(1)) if match else None
        if slurm_job_id is None:
            logger.warning(
                f"Failed to obtain job id for job {self.slurm_name(job_name)}. "
                f"sbatch output: {output}"
            )
            return

        assert isinstance(slurm_job_id, int)
        self.jobs[slurm_job_id] = JobInfo(
            name=self.slurm_name(job_name),
            state=JobState.PENDING,
            slurm_id=slurm_job_id,
        )
        self._update_all()

    def stop(self, job_name, force=False):
        """Stops a running job.

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.

        Args:
            job_name: The job name of the job array to stop.
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`.
        """
        signal = "SIGKILL" if force else "SIGTERM"
        job_id = self.find_job_id(job_name)
        if not job_id:
            return
        return cancel_jobs(slurm_ids=[job_id], signal=signal)

    def stop_all(self, force=False):
        """Stops all running jobs."""
        signal = "SIGKILL" if force else "SIGTERM"
        return cancel_jobs(slurm_ids=list(self.jobs.keys()), signal=signal)

    def find(self, job_name) -> JobInfo | None:
        """Gets the status of a job of this job.

        Args:
            job_name: The job name of the job array to find.
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`.

        Returns:
            A JobInfo if the job is found, or None otherwise.
        """
        self._update_all()
        job_id = self.find_job_id(job_name)
        return self.jobs[job_id] if job_id else None

    def find_all(self, job_name_regex=".*") -> list[JobInfo]:
        """Finds jobs.

        Args:
            job_name_regex: job name regex.

        Returns:
            A list of found JobInfo.
        """
        self._update_all()
        infos = []
        for r in self.jobs.values():
            job_name = r.name.split(":")[-1]  # Extract the job name from slurm name
            if re.fullmatch(job_name_regex, job_name):
                infos.append(r)
        return infos

    def _find_job_with_status(
        self,
        status: list[JobState],
    ) -> list[JobInfo]:
        """Finds jobs with the given status.

        Args:
            status: A list of JobState to filter jobs.

        Returns:
            A list of JobInfo with the given status.
        """
        self._update_all()
        return [r for r in self.jobs.values() if r.state in status]

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
        """Waits until all jobs submitted via this client instance finish."""
        # begin wait
        deadline = None if timeout is None else time.time() + timeout

        num_jobs_left = len(self.jobs)
        left = list(self.jobs.keys())
        logger.info(
            f"Waiting for {num_jobs_left} jobs. Jobs IDs: "
            f"{','.join(sorted([str(x.slurm_id) for x in self.jobs.values()]))}."
        )
        while len(left) > 0:
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {num_jobs_left} jobs. Job ID: "
                    f"{','.join(sorted([str(x.slurm_id) for x in self.jobs.values()]))}."
                )
            self._update_all()
            left = list(self.jobs.keys())
            for slurm_id in list(left):
                slurm_info = self.jobs[slurm_id]
                if slurm_info.slurm_id is None:
                    continue
                if slurm_info.state in check_status:
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=slurm_info.name,
                        host=slurm_info.host,
                        reason=slurm_info.state,
                    )
                if slurm_info.state in remove_status:
                    logger.info(
                        f"Job {slurm_info.name} is {slurm_info.state}. (Removed)"
                    )
                    left.remove(slurm_id)
                    if update:
                        self.jobs.pop(slurm_info.slurm_id)
            time.sleep(SLURM_WAIT_CHECK_TIME_INTERVAL)

    def _update_all(self):
        """Updates the status of all jobs."""
        try:
            slurm_infos = query_jobs(slurm_ids=list(self.jobs.keys()))
            for slurm_info in slurm_infos:
                assert slurm_info.slurm_id is not None
                self.jobs[slurm_info.slurm_id] = slurm_info
        except subprocess.CalledProcessError:
            logger.warning(
                "Calling squeue failed. Check slurm manually if you continue to see this warning."
            )


def main():
    config, _ = parse_cli_args(sys.argv[1:])
    slurm_main(config, run_id=0)


def slurm_main(config, run_id: int = 0):
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    config.recover = to_structured_cfg(config.recover, RecoverConfig)
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
    validate_config_for_distributed_launcher(config)
    logger.info(
        f"SlurmLauncher: experiment_name={config.experiment_name}, "
        f"trial_name={config.trial_name}, fileroot={config.cluster.fileroot}, "
        f"run_id={run_id}, is_recover_run={is_recover_run}"
    )

    # Get scheduling specs from actor config
    actor_spec = get_scheduling_spec(config.actor)

    launcher = SlurmLauncher(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=config.cluster.fileroot,
        container_type=actor_spec.container_type,
    )

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )

    n_nodes = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node
    allocation_mode = config.allocation_mode
    allocation_mode = _AllocationMode.from_str(allocation_mode)

    if not is_recover_run:
        metadata_file = save_experiment_metadata(
            config.cluster.fileroot,
            config.experiment_name,
            config.trial_name,
        )
        logger.info(f"Saved experiment metadata to {metadata_file}")

    n_backend_nodes = 0
    llm_addrs = []

    if allocation_mode.gen_backend in ("sglang", "vllm"):
        # Launcher should launch llm servers according to allocation mode.
        if allocation_mode.gen_backend == "sglang":
            config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
            random_seed = config.sglang.random_seed
        else:
            config.vllm = to_structured_cfg(config.vllm, vLLMConfig)
            random_seed = config.vllm.seed

        config.rollout = to_structured_cfg(config.rollout, InferenceEngineConfig)
        # Get rollout scheduling spec
        rollout_spec = get_scheduling_spec(config.rollout)

        backend_spec = {
            "sglang": {
                "module": "areal.infra.launcher.sglang_server",
                "seed_arg": "sglang.random_seed",
                "prefix": "AREAL_SGLANG",
                "set_device_env": False,
            },
            "vllm": {
                "module": "areal.infra.launcher.vllm_server",
                "seed_arg": "vllm.seed",
                "prefix": "AREAL_VLLM",
                "set_device_env": True,  # vLLM needs `device_control_env_var` to control GPU allocation
            },
        }

        def _build_llm_server_plan(backend: str, spec: dict):
            # Returns: cmds, env_vars_list, n_nodes, n_servers

            if backend not in backend_spec:
                raise NotImplementedError(f"Unknown backend: {backend}")

            spec = backend_spec[backend]

            n_backend_servers = allocation_mode.gen.dp_size
            n_backend_nodes = allocation_mode.gen.world_size // n_gpus_per_node
            node_group_size = max(
                1, allocation_mode.gen_instance_size // n_gpus_per_node
            )
            n_servers_per_node = max(n_backend_servers // n_backend_nodes, 1)

            cross_nodes = allocation_mode.gen_instance_size > n_gpus_per_node
            rollout_cpus_per_task = rollout_spec.cpu * n_gpus_per_node
            thread_env = get_thread_env_vars(
                cpus_per_task=rollout_cpus_per_task,
                existing_env_vars=rollout_spec.env_vars,
            )
            base_env_vars = {**BASE_ENVIRONS, **thread_env, **rollout_spec.env_vars}
            if spec["set_device_env"]:
                base_env_vars[current_platform.device_control_env_var] = ",".join(
                    list(map(str, range(n_gpus_per_node)))
                )
            env_list = [copy.deepcopy(base_env_vars) for _ in range(n_backend_nodes)]

            base_seed = random_seed
            seed_arg = spec["seed_arg"]
            module = spec["module"]
            backend_server_cmd_template = (
                f"python3 -m {module} {' '.join(sys.argv[1:])} {seed_arg}={{seed}}"
            )

            backend_cmds = []
            for i in range(n_backend_nodes):
                backend_cmd = backend_server_cmd_template.format(
                    seed=base_seed + i * n_servers_per_node
                )
                backend_cmds.append(backend_cmd)
                if cross_nodes:
                    # master_addrs and master_ports are the IP addresses and free ports of the all nodes in the job array, obtained in the SBATCH script.
                    prefix = spec["prefix"]
                    env_list[i] |= dict(
                        **{
                            f"{prefix}_MULTI_NODE_RANK": i % node_group_size,
                            f"{prefix}_MULTI_NODE_MASTER_ADDR": f"${{master_addrs[{i // node_group_size * node_group_size}]}}",
                            f"{prefix}_MULTI_NODE_MASTER_PORT": f"${{master_ports[{i // node_group_size * node_group_size}]}}",
                        }
                    )

            return backend_cmds, env_list, n_backend_nodes, n_backend_servers

        backend_cmds, env_list, n_backend_nodes, n_backend_servers = (
            _build_llm_server_plan(
                allocation_mode.gen_backend,
                random_seed,
            )
        )

        launcher.submit_array(
            job_name="llm_server",
            cmd=backend_cmds,
            count=n_backend_nodes,
            nodes=n_backend_nodes,
            n_gpus_per_node=n_gpus_per_node,
            cpus_per_task=rollout_spec.cpu * n_gpus_per_node,
            mem_per_task=rollout_spec.mem * 1024 * n_gpus_per_node,
            srun_additional_args=rollout_spec.srun_additional_args,
            container_image=rollout_spec.image,
            container_mounts=rollout_spec.mount,
            env_vars=env_list,
        )
        # Get llm server addresses by name resolve
        try:
            llm_addrs = wait_llm_server_addrs(
                config.experiment_name,
                config.trial_name,
                n_backend_servers,
            )
        except (TimeoutError, KeyboardInterrupt) as e:
            launcher.stop_all(force=True)
            raise e

    trainer_n_nodes = n_nodes - n_backend_nodes
    gpus_per_node = config.cluster.n_gpus_per_node

    def _build_trainer_cmds(
        trainer_n_nodes: int,
        nproc_per_node: int,
        additional_bash_cmds: list[str] | None = None,
    ) -> list[str]:
        extra_args = " ".join(sys.argv[1:])
        trainer_cmds = []
        for i in range(trainer_n_nodes):
            # In slurm, we launch trainer in the granularity of nodes with torchrun command.
            bash_cmds = (additional_bash_cmds or []).copy()
            # Here $head_node_ip is the IP address of the first node in the job array.
            # $trainer_port is a free port on the head node.
            # Both of them are obtained in by the SBATCH script.
            torchrun_cmd = " ".join(
                [
                    "torchrun",
                    f"--nnodes={trainer_n_nodes}",
                    f"--nproc-per-node={nproc_per_node}",
                    f"--node-rank={i}",
                    "--master-addr=$head_node_ip",
                    "--master-port=$trainer_port",
                    extra_args,
                ]
            )
            bash_cmds.append(torchrun_cmd)
            bash_cmds_str = ";\n".join(bash_cmds)
            # handle double quotes in bash commands
            bash_cmds_str = bash_cmds_str.replace('"', '\\"')
            trainer_cmds.append(f'bash -c "{bash_cmds_str}"')

        if trainer_n_nodes:
            logger.info(
                f"Trainer commands for the first trainer node:\n{trainer_cmds[0]}"
            )
        else:
            logger.warning("No trainer commands")
        return trainer_cmds

    if config.get("enable_offload", False):
        tms_env_vars = get_tms_env_vars()
    else:
        tms_env_vars = {}

    if allocation_mode.type_ != AllocationType.LLM_SERVER_ONLY:
        # launch trainers
        _env_vars = dict(
            AREAL_LLM_SERVER_ADDRS=",".join(llm_addrs),
        )
        if allocation_mode.gen_backend == "sglang":
            # Required by NCCL weight update group.
            _env_vars["NCCL_CUMEM_ENABLE"] = "0"
            _env_vars["NCCL_NVLS_ENABLE"] = "0"

        trainer_cpus_per_task = actor_spec.cpu * config.cluster.n_gpus_per_node
        # Use per-GPU CPU count for thread env vars since torchrun spawns
        # n_gpus_per_node processes, each inheriting these env vars.
        # This matches Local and Ray launcher behavior.
        thread_env = get_thread_env_vars(
            cpus_per_task=actor_spec.cpu,
            existing_env_vars=actor_spec.env_vars,
        )
        launcher.submit_array(
            job_name="trainer",
            cmd=_build_trainer_cmds(
                trainer_n_nodes,
                config.cluster.n_gpus_per_node,
                actor_spec.additional_bash_cmds,
            ),
            count=trainer_n_nodes,
            nodes=trainer_n_nodes,
            n_gpus_per_node=gpus_per_node,
            cpus_per_task=trainer_cpus_per_task,
            mem_per_task=actor_spec.mem * 1024 * config.cluster.n_gpus_per_node,
            container_image=actor_spec.image,
            srun_additional_args=actor_spec.srun_additional_args,
            container_mounts=actor_spec.mount,
            env_vars={
                **BASE_ENVIRONS,
                **thread_env,
                **actor_spec.env_vars,
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
        launcher.stop_all(force=True)
        recover_states = [JobState.FAILED, JobState.NOT_FOUND]
        if isinstance(e, JobException):
            recover_this = (
                e.reason in recover_states
                and run_id < config.recover.retries
                and config.recover.mode in ("on", "auto")
            )
            if recover_this:
                time.sleep(RECOVER_TIME_INTERVAL)
                slurm_main(config, run_id=run_id + 1)
            else:
                raise e
        else:
            raise e


if __name__ == "__main__":
    # usage: python -m areal.infra.launcher.slurm <entry_point> \
    #  --config <config_path> [<additional_args>]
    main()
