import os
from pypai.job import PythonJobBuilder
from pypai.conf import ExecConf, KMConf
from pypai.conf.retry_strategy import RetryStrategy, RetryPolicy
from aistudio_common.openapi.models.data_store import DataStore

worker_num = int(os.environ.get('WORKER_NUM', '0'))
gpu_num = int(os.environ.get('GPU_NUM', '5'))
rs = RetryStrategy(retry_policy=RetryPolicy.ON_EVICTION, max_attempt=2)
master = ExecConf(cpu=16*gpu_num, memory=192000*gpu_num, disk_m=200000*gpu_num, gpu_num=gpu_num, num=1, gpu_type='h200')
worker = ExecConf(cpu=16*gpu_num, memory=192000*gpu_num, disk_m=200000*gpu_num, gpu_num=gpu_num, num=worker_num, gpu_type='h200')
km_conf = KMConf(image=os.environ.get('KM_IMAGE', 'acr-sh-ant-registry-vpc.cn-shanghai.cr.aliyuncs.com/gpu/areal-runtime:dev-sglang-20260401'), retry_strategy=rs)
command = os.environ['JOB_COMMAND']
job = PythonJobBuilder(source_root=None, main_file='', name=os.environ.get('JOB_NAME') or None, command=command, km_conf=km_conf, k8s_app_name='areals3', k8s_priority='low', master=master, worker=worker, host_network=True, runtime='pytorch', rdma=True, tag=os.environ.get('JOB_TAG',''), enable_pcache_fuse=True, data_stores=[DataStore(mount_point='/storage/', store_name='s3-asys-cpfs', sub_path='/o1/')])
job.run(enable_wait=False)
