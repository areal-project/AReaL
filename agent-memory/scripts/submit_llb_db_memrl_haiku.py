import os
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore

# LLB DB 任务只需 API 调用，不需要 GPU 计算，但必须指定 h200 调度到 et15-aidc（有 /storage）
master = ExecConf(cpu=16, memory=64000, disk_m=100000, gpu_num=1, num=1, gpu_type="h200")
worker = ExecConf(cpu=16, memory=64000, disk_m=100000, gpu_num=1, num=0, gpu_type="h200")

# ON_EVICTION covers platform preemption (which ON_FAILURE does NOT). Together
# with a stable MEMRL_RUN_ID in the command, a preempted job auto-restarts and
# resumes from the latest snapshot instead of starting over from S1.
rs = RetryStrategy(retry_policy=RetryPolicy.ON_EVICTION, max_attempt=10)
km_conf = KMConf(
    image="acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401",
    retry_strategy=rs,
)

import time as _time
_default_job_name = f"yl-llb-db-memrl-haiku-{_time.strftime('%m%d-%H%M')}"
job_name = os.environ.get("JOB_NAME", _default_job_name) or _default_job_name
command = os.environ.get(
    "JOB_COMMAND",
    "bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_memrl_haiku_aistudio.sh",
)


def gpujob():
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
        data_stores=[
            DataStore(mount_point="/storage/", store_name="s3-asys-cpfs", sub_path="/o1/"),
        ],
    )
    job.run(enable_wait=False)


if __name__ == "__main__":
    gpujob()
