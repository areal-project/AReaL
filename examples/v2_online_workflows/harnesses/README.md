# Agent Harness

`hermes.py` 提供 Hermes Agent Service 适配，`cc.py` 提供 Claude Agent SDK 适配；
`hermes_runtime.py` 负责可复用的工具绑定、模型调用观察与 episode 身份， `protocol_adapter.py` 复用 AReaL
已有的协议转换。任务组织由上层 `service.py` 完成。

这里保留与具体发布平台无关的 Agent 实现。使用方式见 [工作流说明](../README.md)。 Arena
的沙箱入口、工具适配、独立评分入口、运行包和联调脚本属于本地部署物料，均放在工作区 local 中。
