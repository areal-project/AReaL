# Elastic 八卡功能检查

正常 demo 使用 `python -m examples.elastic.reproduce`。本目录保留两项显式功能验收，参数与边界见
[VALIDATION.md](../../examples/elastic/VALIDATION.md)。

| 模块                      | 用途                             |
| ------------------------- | -------------------------------- |
| `elastic_gpu_restart`     | 三步训练中的强制释放、恢复与续训 |
| `elastic_gpu_weight_sync` | TP4→两组 TP2 的权重发布核验      |

在仓库根目录执行 `python -m tests.elastic_tools.<模块> --help`。旧
`tests.elastic_gpu_restart`／`tests.elastic_gpu_weight_sync` 命令继续转发。

保存／恢复计时与恢复路径 A/B 实验已移至项目外层 `experiments/elastic_lifecycle/`，不属于本目录或 demo 的运行依赖。
