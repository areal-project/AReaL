# SPDX-License-Identifier: Apache-2.0

import subprocess
from typing import Literal

from areal.infra.utils.launcher import (
    JobInfo,
    JobState,
)
from areal.utils import logging

logger = logging.getLogger("SlurmUtils")


SQUEUE_FIELDS = [
    "JobID",
    "State",
    "SubmitTime",
    "StartTime",
    "Name",
    "NodeList",
    "UserName",
    "MaxCPUs",
    "cpus-per-task",
    "NumTasks",
    "tres-alloc",
]
STATUS_MAPPING = {
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "PENDING": JobState.PENDING,
    "CANCELLED": JobState.CANCELLED,
    "FAILED": JobState.FAILED,
    "COMPLETED": JobState.COMPLETED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "DEADLINE": JobState.COMPLETED,
    "TIMEOUT": JobState.COMPLETED,
}

SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
{sbatch_options}

##### Setup failure capture and clean up #####
# Array to track background PIDs
declare -a bg_pids=()

# Function to clean up background processes
cleanup_bg_jobs() {{
    for pid in "${{bg_pids[@]}}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing background job $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait a bit for processes to terminate
    sleep 0.5
    # Force kill if still running
    for pid in "${{bg_pids[@]}}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}}

# Trap to ensure cleanup on exit
trap cleanup_bg_jobs EXIT

##### Get IP addresses and submit jobs #####

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo nodes=$nodes

nodes_array=($nodes)
echo node_array=$nodes_array

head_node=${{nodes_array[0]}}
echo head_node=$head_node

# Getting the head node IP address
head_node_ip=$(srun {srun_additional_args} --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist="$head_node" hostname --ip-address)
echo head_node_ip=$head_node_ip

# Find a free port on the head node
# Wonderful linux command to find a random free port (between 10000 and 60000) by deepseek
trainer_port=$(srun {srun_additional_args} --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist="$head_node" bash -c "comm -23 <(seq 10000 60000 | sort) <(ss -tan | awk '{{print $4}}' | cut -d':' -f2 | grep '[0-9]\\{{1,5\\}}' | sort -u) | shuf | head -n 1")
echo trainer_port=$trainer_port

# Get IP address of each node
master_addrs=()
for node in "${{nodes_array[@]}}"; do
    ip=$(srun {srun_additional_args} --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist="$node" hostname --ip-address)
    master_addrs+=("$ip")
done
echo master_addrs="${{master_addrs[@]}}"
# Get a free port for each node
master_ports=()
for node in "${{nodes_array[@]}}"; do
    port=$(srun {srun_additional_args} --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist="$node" bash -c "comm -23 <(seq 10000 60000 | sort) <(ss -tan | awk '{{print $4}}' | cut -d':' -f2 | grep '[0-9]\\{{1,5\\}}' | sort -u) | shuf | head -n 1")
    master_ports+=("$port")
done

# srun commands
{srun_cmds}

##### Monitor all processes #####
declare -a still_running=()
while [ ${{#bg_pids[@]}} -gt 0 ]; do
    unset still_running
    for i in "${{!bg_pids[@]}}"; do
        pid=${{bg_pids[$i]}}
        if ! kill -0 "$pid" 2>/dev/null; then
            # Process has terminated, check its exit code
            if wait "$pid"; then
                # Process completed successfully
                echo "Process $pid completed successfully"
            else
                echo "Process $pid failed, terminating remaining jobs"

                exit 1  # This will trigger cleanup via trap
            fi
        else
            still_running+=("$pid")
        fi
    done
    bg_pids=("${{still_running[@]}}")

    # Break if no processes left
    if [ ${{#bg_pids[@]}} -eq 0 ]; then
        break
    fi

    sleep 0.1  # Small delay to avoid busy waiting
done
"""

SRUN_CMD_TEMPLATE: str = """srun {additional_args} \\
    --nodelist=${{nodes_array[{node_id}]}} --nodes={nodes} --ntasks={ntasks} \\
    --gres=gpu:{n_gpus_per_node} --cpus-per-task={cpus_per_task} --mem-per-cpu={mem_per_cpu}M \\
    {cmd} &
bg_pids+=($!)

"""

APPTAINER_CMD_TEMPLATE: str = """singularity exec --no-home --writable-tmpfs --nv --pid \\
    --bind {container_mounts} \\
    {container_env_strings} \\
    {container_image} \\
    {cmd}"""


def cancel_jobs(
    slurm_names: list[str] | None = None,
    slurm_ids: list[int] | None = None,
    signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL",
):
    if slurm_names is None and slurm_ids is None:
        raise RuntimeError("Must specify slurm_names or slurm_ids.")
    if slurm_names and slurm_ids:
        raise RuntimeError("Cannot specify both slurm_names and slurm_ids.")
    cmd = ["scancel", "-s", signal]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    elif slurm_ids is not None:
        cmd += [",".join(str(s) for s in slurm_ids)]
    try:
        subprocess.check_call(cmd)
        logger.info(
            f"Cancelled Slurm job with signal {signal}: "
            f"slurm identifiers {slurm_names if slurm_ids is None else slurm_ids}. CMD: {cmd}"
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Cancel slurm job failed, reason: {e}")


def query_jobs(
    slurm_names: list[str] | None = None,
    slurm_ids: list[int] | None = None,
    status: str = "all",
    delimiter: str = "__PSI__",
) -> list[JobInfo]:
    squeue_format = f":.{delimiter},".join(SQUEUE_FIELDS)
    cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    if slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]

    output = (
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    )
    rs = []
    for line in output.split("\n")[1:]:
        job_id, state, submit_time, start_time, slurm_name, nodelist, *_ = line.split(
            delimiter
        )
        rs.append(
            JobInfo(
                name=slurm_name,
                state=STATUS_MAPPING[state],
                host=nodelist,
                submit_time=submit_time,
                start_time=start_time,
                slurm_id=int(job_id.strip()),
            )
        )
    return rs


def parse_slurm_nodelist(nodelist: str) -> list[str]:
    return (
        subprocess.check_output(
            [
                "scontrol",
                "show",
                "hostnames",
                nodelist,
            ]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )


def get_slurm_host_ip(node: str, srun_addtional_args: str):
    try:
        cmd = f"srun {srun_addtional_args} --immediate=1 --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist={node} hostname --ip-address"
        return subprocess.check_output(cmd.split(" ")).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        logger.warning(f"Get slurm host IP for node {node} failed.")
