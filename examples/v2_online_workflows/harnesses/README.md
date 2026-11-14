# Hermes 运行层

`service.py` 维护常驻 worker 池，`worker.py` 为每个请求准备独立工具环境并运行 `episode.py`。 `executor.py`
保留请求身份，调用服务后直接读取 answer、environment 和 call trace。 `hermes_runtime.py` 绑定采样参数和 AReaL
episode，`tools.py` 接入 Hermes 原生 terminal/file 工具。 工具隔离实现位于
`../environments/hermes.py`，调用回执和字段与原本地运行层兼容。
