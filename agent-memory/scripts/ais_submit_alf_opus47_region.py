"""AIStudio submit template for ALFWorld opus-4-7 Region+FS."""
import os

from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore

app_name = "areals3"
worker_num = int(os.environ.get("WORKER_NUM", "0"))
job_name = os.environ.get("JOB_NAME", "") or None

retry_policy_name = os.environ.get("AIS_RETRY_POLICY", "ON_FAILURE").strip().upper()
retry_policy = getattr(RetryPolicy, retry_policy_name)
retry_max_attempt = int(os.environ.get("AIS_RETRY_MAX_ATTEMPT", "3"))
rs = RetryStrategy(retry_policy=retry_policy, max_attempt=retry_max_attempt)

# Only need 1 GPU for scheduling to et15-aidc (h200), actual compute is API-only
master = ExecConf(cpu=16, memory=64000, disk_m=100000, gpu_num=1, num=1, gpu_type="h200")
worker = ExecConf(cpu=16, memory=64000, disk_m=100000, gpu_num=1, num=worker_num, gpu_type="h200")
km_conf = KMConf(
    image=os.environ.get(
        "KM_IMAGE",
        "acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401",
    ),
    retry_strategy=rs,
)

command = os.environ.get("JOB_COMMAND", "")
if not command:
    raise ValueError("JOB_COMMAND 环境变量未设置")


def gpujob():
    job = PythonJobBuilder(
        source_root=None,
        main_file="",
        name=job_name,
        command=command,
        km_conf=km_conf,
        k8s_app_name=app_name,
        k8s_priority="low",
        master=master,
        worker=worker,
        host_network=True,
        runtime="pytorch",
        rdma=False,
        tag=os.environ.get("JOB_TAG", ""),
        enable_pcache_fuse=True,
        data_stores=[
            DataStore(mount_point="/storage/", store_name="s3-asys-cpfs", sub_path="/o1/"),
        ]
    )
    job.run(enable_wait=False)


if __name__ == "__main__":
    gpujob()
