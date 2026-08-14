#!/usr/bin/env python3
"""Submit LLB-OS Mem0 GPT-4.1-mini and atomically register its AIS record."""
import os
import sys
from pathlib import Path

from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from llb_os_job_guard import preflight, register  # noqa: E402

master = ExecConf(cpu=16, memory=64000, disk_m=100000, gpu_num=1, num=1, gpu_type="h200")
worker = ExecConf(cpu=16, memory=64000, disk_m=100000, gpu_num=1, num=0, gpu_type="h200")
km_conf = KMConf(
    image=os.environ.get("KM_IMAGE", "acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401"),
    retry_strategy=RetryStrategy(retry_policy=RetryPolicy.ON_EVICTION, max_attempt=10),
)


def main() -> None:
    task_key = os.environ["LLB_OS_TASK_KEY"]
    job_name = os.environ["JOB_NAME"]
    command = os.environ["JOB_COMMAND"]
    preflight(task_key)
    job = PythonJobBuilder(
        source_root=None,
        main_file="",
        name=job_name,
        command=command,
        km_conf=km_conf,
        k8s_app_name="areals3",
        k8s_priority="low",
        master=master,
        worker=worker,
        host_network=True,
        runtime="pytorch",
        rdma=True,
        tag=os.environ.get("JOB_TAG", ""),
        enable_pcache_fuse=True,
        data_stores=[DataStore(mount_point="/storage/", store_name="s3-asys-cpfs", sub_path="/o1/")],
    )
    record_id = job.run(enable_wait=False)
    if not record_id:
        raise RuntimeError("AIS submission returned no record_id")
    print(f"[AIS] submitted job_name={job_name} record_id={record_id}")
    register(task_key, str(record_id), job_name)


if __name__ == "__main__":
    main()
