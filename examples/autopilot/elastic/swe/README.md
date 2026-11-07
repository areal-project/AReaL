# SWE 沙箱工具

本目录提供 AEnvironment 沙箱部署配置、fixture 准备、评分检查及轨迹转换工具。使用 Linux amd64、Python 3.12 和已部署的
AEnvironment；集群部署步骤见外层[SWE 操作说明](../../../../../docs/runbooks/swe-sandbox.md)。

## 准备客户端

设置各 checkout 与数据目录的绝对路径：

```bash
export AREAL_ROOT=/path/to/AReaL
export AENV_ROOT=/path/to/AEnvironment
export SWE_AGENT_ROOT=/path/to/AReaL-SWEAgent
export PROJECT_ROOT=/path/to/project-data
export SETUP="$AREAL_ROOT/examples/autopilot/elastic/swe"
export KUBECONFIG="$PROJECT_ROOT/kind/kubeconfig"

uv venv --python python3.12 "$PROJECT_ROOT/swe/client-venv"
uv --no-config pip install --python "$PROJECT_ROOT/swe/client-venv/bin/python" \
  -e "$AENV_ROOT/aenv" pydantic==2.11.10 loguru
export PYTHONPATH="$AREAL_ROOT:$SWE_AGENT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export AENV_SYSTEM_URL=http://127.0.0.1
export LOG_DIR="$PROJECT_ROOT/logs/swe"
```

AEnvironment 与训练客户端使用独立 Python 环境，通过 JSON-lines RPC 通信。

## 检查 fixture 与评分

准备同一任务的 row JSON 和 grading details JSON，运行：

```bash
python3 "$SETUP/prepare_fixture.py" --row /path/to/row.json \
  --details /path/to/details.json --output "$PROJECT_ROOT/swe/fixture.json"

"$PROJECT_ROOT/swe/client-venv/bin/python" -m examples.autopilot.elastic.swe.smoke_sandbox \
  --fixture "$PROJECT_ROOT/swe/fixture.json" --environment autopilot-swe-bash@0.1.0 \
  --report "$PROJECT_ROOT/swe/smoke-report.json" \
  --kubectl "$(command -v kubectl)" --kubeconfig "$KUBECONFIG"
```

该命令检查空 patch／参考 patch、持久 shell、任务评分与沙箱清理，不启动模型或训练。数据来源、固定依赖版本与既有验收记录见操作说明。

原生 DAPO／SWE 任务提交使用[共享任务工作流](../../../task_workflows/README.md)。
