import os

from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf
from pypai.conf import KMConf
from pypai.conf import GpuType
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore

'''
BCB 双节点专用提交模板(yl 自维护, 不改动 public 的 submit_template.py)。

与 public 模板的唯一区别: master 的 gpu_num 可由 MASTER_GPU_NUM 控制。
  - 双节点 BCB: worker 8 卡跑 DeepSeek-V3 TP=8 主 LLM; master 只需 1 卡跑
    embedding(独占) + run_bcb, 设 MASTER_GPU_NUM=1 省掉 7 张闲置卡。
  - 单节点(WORKER_NUM=0): master 8 卡, MASTER_GPU_NUM 默认 8。

单位: cpu=核, memory=MB, gpu_num=卡数, num=节点数。可调参数走环境变量(见 submit.sh)。
'''
app_name = "areals3"
worker_num = int(os.environ.get("WORKER_NUM", "0"))
master_gpu_num = int(os.environ.get("MASTER_GPU_NUM", "8"))
job_name = os.environ.get("JOB_NAME", "") or None

rs = RetryStrategy(retry_policy=RetryPolicy.ON_FAILURE, max_attempt=int(os.environ.get("AIS_MAX_ATTEMPT", "5")))

master = ExecConf(cpu=96, memory=500000, disk_m=1000000, gpu_num=master_gpu_num, num=1, gpu_type="h200")
worker = ExecConf(cpu=96, memory=500000, disk_m=1000000, gpu_num=8, num=worker_num, gpu_type="h200")
km_conf = KMConf(
    image=os.environ.get(
        "KM_IMAGE",
        "acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401",
    ),
    retry_strategy=rs,
)

os.chdir("/storage/openpsi/users/yl/agent-memory/MemRL/scripts")  # writable workflow staging dir

command = os.environ.get("JOB_COMMAND", "")
if not command:
    raise ValueError("JOB_COMMAND 环境变量未设置或为空，请在 submit.sh 中指定启动命令")


def gpujob():
    job = PythonJobBuilder(
        source_root=None,
        main_file=__file__,
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
