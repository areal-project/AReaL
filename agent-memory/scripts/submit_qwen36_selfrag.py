import os

from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf
from pypai.conf import KMConf
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore

app_name = "areals3"
worker_num = int(os.environ.get("WORKER_NUM", "0"))
job_name = os.environ.get("JOB_NAME", "") or None

rs = RetryStrategy(retry_policy=RetryPolicy.ON_FAILURE, max_attempt=1)

gpu_num = int(os.environ.get("GPU_NUM", "2"))
master = ExecConf(cpu=32, memory=400000, disk_m=500000, gpu_num=gpu_num, num=1, gpu_type="h200")
worker = ExecConf(cpu=32, memory=400000, disk_m=500000, gpu_num=gpu_num, num=worker_num, gpu_type="h200")
km_conf = KMConf(
    image=os.environ.get(
        "KM_IMAGE",
        "acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-vllm-20260429",
    ),
    retry_strategy=rs,
)

command = os.environ.get("JOB_COMMAND", "")
if not command:
    raise ValueError("JOB_COMMAND 环境变量未设置或为空")


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
        rdma=True,
        tag=os.environ.get("JOB_TAG", ""),
        enable_pcache_fuse=True,
        data_stores=[
            DataStore(mount_point="/storage/", store_name="s3-asys-cpfs", sub_path="/o1/"),
        ]
    )
    job.run(enable_wait=False)


if __name__ == "__main__":
    gpujob()
