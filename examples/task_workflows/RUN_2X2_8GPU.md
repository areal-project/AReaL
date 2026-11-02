# Hermes／CC × DAPO／SWE：八卡评测与常驻训练

适用于单机8×H20或8×H200。四种组合共用 Qwen3-1.7B、v2 推理链路和现有 online PPO；DAPO 在这里指数学数据集。本文不开
on_demand，不修改算法，也不要求生成结果逐 token 相同。

| CASE        | Harness     | Benchmark      | Agent 类    | 配置      |
| ----------- | ----------- | -------------- | ----------- | --------- |
| hermes-dapo | Hermes      | DAPO 数学      | TaskAgent   | dapo.yaml |
| cc-dapo     | Claude Code | DAPO 数学      | CCTaskAgent | dapo.yaml |
| hermes-swe  | Hermes      | SWE-rebench V2 | TaskAgent   | swe.yaml  |
| cc-swe      | Claude Code | SWE-rebench V2 | CCTaskAgent | swe.yaml  |

**验证边界**：共享适配21项离线检查通过；CC 的真实 SDK／CLI 已通过替身模型和 Docker 传输验证，尚未通过真实 GPU 训练。旧 Hermes
数学16题／4更新已通过，正式 batch40 在第4次更新 OOM；新组合及 H200 均待实测。下文是待执行步骤，不代表四组已经验收。

## 1．两种机器的公共准备

| 项目             | 8×H20                            | 8×H200                      |
| ---------------- | -------------------------------- | --------------------------- |
| 推理评测布局     | TP1×8副本                        | 相同，先保持可比            |
| 常驻训练布局     | FSDP4 Actor＋4个TP1 Rollout      | 相同，先保持可比            |
| 起始模型／上下文 | 1.7B／BF16／64K YaRN／非thinking | 使用同一模型目录内容        |
| Rollout 显存比例 | 0.5，作为起点                    | 先用0.5，后续调优单独记配置 |
| GPU 前提         | 八卡均已分配且空闲               | 使用公司实际分配的八卡      |

机器差异先只体现在硬件标签和工作目录。64K是上下文容量，不是每次都生成64K；单次最多生成4096 token。SWE 还依赖 CPU、内存和 Docker
镜像空间，不能只看显存。

在服务器宿主机设置下列路径；H200设 `TASK_HW=h200`，换成公司允许使用的路径和兼容的 AReaL CUDA／SGLang 镜像。现有 H20 容器准备细节可参考
[EVAL_8H20.md](EVAL_8H20.md)。

```bash
export TASK_HW=h20
export TASK_HOST_ROOT=/absolute/path/to/task-workflows-2x2
export TASK_CODE_SOURCE=/absolute/path/to/current/AReaL
export TASK_MODEL_SOURCE=/absolute/path/to/original/Qwen3-1.7B
export TASK_DAPO_SOURCE=/absolute/path/to/frozen/dapo-tasks
export TASK_CONTAINER="task-workflows-2x2-$TASK_HW"
export TASK_RUNTIME_IMAGE=ghcr.io/areal-project/areal-runtime:v2.1.0-sglang

mkdir -p "$TASK_HOST_ROOT"
nvidia-smi
docker version
docker info --format '{{.DockerRootDir}}'
df -h "$TASK_HOST_ROOT"
```

镜像存储所在文件系统也要检查。公司端首次取代码可直接 clone 本分支；已有 checkout 则在该 checkout 执行
`git pull --ff-only origin feature/autopilot-elastic`：

```bash
git clone --branch feature/autopilot-elastic \
  git@github.com:Tournesoly/areal-autopilot-elastic.git "$TASK_CODE_SOURCE"
```

将代码和**同一份冻结题单**放到工作目录；模型和题单不随 Git 分发。源／目标目录须不同：

```bash
mkdir -p "$TASK_HOST_ROOT/AReaL" "$TASK_HOST_ROOT/tasks/dapo" "$TASK_HOST_ROOT/models"
rsync -a --exclude=.git --exclude=.venv --exclude=__pycache__ \
  --exclude=.pytest_cache "$TASK_CODE_SOURCE/" "$TASK_HOST_ROOT/AReaL/"
rsync -a "$TASK_DAPO_SOURCE/" "$TASK_HOST_ROOT/tasks/dapo/"
test ! -e "$TASK_HOST_ROOT/models/qwen3-17b-64k" || exit 1
rsync -aL "$TASK_MODEL_SOURCE/" "$TASK_HOST_ROOT/models/qwen3-17b-64k/"

docker pull "$TASK_RUNTIME_IMAGE"
docker pull python:3.12
docker run -dit --name "$TASK_CONTAINER" --gpus all \
  --network host --shm-size 16g \
  -v "$TASK_HOST_ROOT:/workspace" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(command -v docker):/usr/local/bin/docker:ro" \
  -w /workspace/AReaL -e PYTHONPATH=/workspace/AReaL \
  -e TASK_HW="$TASK_HW" \
  "$TASK_RUNTIME_IMAGE" sleep infinity

docker exec -it "$TASK_CONTAINER" bash
```

看到容器提示符后，再粘贴后续命令。以下步骤均在这个容器内；工具和评分通过同一 Docker daemon 创建任务容器。

```bash
uv pip install --python "$(command -v python)" --no-deps -e .
uv pip install --python "$(command -v python)" \
  'hermes-agent==0.19.0' 'claude-agent-sdk==0.1.71' 'mcp==1.26.0' 'math-verify==0.8.0'
mkdir -p /workspace/results
uv pip freeze --python "$(command -v python)" > /workspace/results/packages.txt

umask 077
cat > /workspace/2x2.env <<EOF
export TASK_HW=$TASK_HW
export AREAL_MODEL_PATH=/workspace/models/qwen3-17b-64k
export AREAL_INFERENCE_ADMIN_KEY=$(openssl rand -hex 24)
export AREAL_AGENT_ADMIN_KEY=$(openssl rand -hex 24)
export AREAL_TASK_PYTHON_IMAGE=python:3.12
export AREAL_TASK_MAX_TOKENS=4096
export AREAL_TASK_TEMPERATURE=1.0
export AREAL_TASK_TOP_P=1.0
export AREAL_SWE_EVALUATOR_ROOT=/workspace/swe-evaluator
EOF
source /workspace/2x2.env
```

环境文件含密钥，留在本机，不提交或粘贴。以后每个新容器终端先执行 `source /workspace/2x2.env`。两台机器保留相同的软件版本；不要为 H200
临时升级整套依赖后直接混合比较。

### 准备共同模型

仅修改上面新复制的模型配置，不改原始权重。原模型须为未扩展的 Qwen3-1.7B：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["AREAL_MODEL_PATH"])
config = json.loads((root / "config.json").read_text())
assert config["max_position_embeddings"] == 32768
config["max_position_embeddings"] = 65536
config["rope_scaling"] = {
    "rope_type": "yarn", "factor": 2.0,
    "original_max_position_embeddings": 32768, "rope_theta": 1000000,
}
tokenizer = json.loads((root / "tokenizer_config.json").read_text())
assert isinstance(tokenizer["chat_template"], str)
tokenizer["chat_template"] = "{% set enable_thinking = false %}\n" + tokenizer["chat_template"]
tokenizer["model_max_length"] = 65536
for name, value in (("config.json", config), ("tokenizer_config.json", tokenizer)):
    (root / name).write_text(json.dumps(value, indent=2) + "\n")
PY
sha256sum "$AREAL_MODEL_PATH"/*.safetensors \
  "$AREAL_MODEL_PATH/config.json" "$AREAL_MODEL_PATH/tokenizer_config.json" \
  > /workspace/results/model-sha256.txt
python -m examples.task_workflows.environments.python
```

最后应输出42且执行成功。模型配置和 tokenizer 同时设64K，是因为 v2 在代理侧应用聊天模板；只给裸 SGLang
加模板参数不够。两台机器应复制这一份准备好的模型并核对摘要。

### 冻结任务与 SWE 环境

DAPO 直接复用已有 `train.jsonl、eval.jsonl、manifest.json`；已有200题 eval 的 SHA256为
`e899346883efa7bd1f4cf3abb8319684bbbcd2deb2e397b3e3f8ca54640f1c49`。不要在不同组合／硬件上重新抽题。

首次准备 SWE：先选32个 Python 任务，按固定顺序分成训练16题和评测16题。它们的 instance ID 不重叠，不代表仓库级隔离。两个 harness
和两种硬件复用同一份清单。

```bash
python -m examples.task_workflows.benchmarks.swe_rebench prepare \
  --output /workspace/tasks/swe-pool --language python --count 32 --seed 1
python -m examples.task_workflows.benchmarks.swe_rebench pull-images \
  --tasks /workspace/tasks/swe-pool/tasks.jsonl
python - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
from examples.task_workflows.benchmarks.swe_rebench import load_tasks, write_tasks
root = Path("/workspace/tasks")
tasks, manifest = load_tasks(root / "swe-pool/tasks.jsonl")
records = [dict(task.record) for task in tasks]
for record in records:
    image = record["image_name"]
    digests = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        text=True))
    assert digests, f"No registry digest for {image}"
    record["image_name"] = sorted(digests)[0]
source = {"parent_manifest_sha256": hashlib.sha256(
    (root / "swe-pool/manifest.json").read_bytes()).hexdigest(),
    "parent_source": manifest["source"], "split": "ordered_first16_rest16"}
write_tasks(records[:16], root / "swe-train", source)
write_tasks(records[16:], root / "swe-eval", source)
PY
git clone https://github.com/SWE-rebench/SWE-rebench-V2 "$AREAL_SWE_EVALUATOR_ROOT"
git -C "$AREAL_SWE_EVALUATOR_ROOT" checkout c71902a8cf8d2b725f63d51f199f4d3e56f68d2d
set -o pipefail
for split in train eval; do
  python -m examples.task_workflows.benchmarks.swe_rebench grade-reference \
    --tasks "/workspace/tasks/swe-$split/tasks.jsonl" \
    --output "/workspace/results/reference-$split" \
    | tee "/workspace/results/reference-$split.jsonl" || break
done
python - <<'PY'
import json
from pathlib import Path
for split in ("train", "eval"):
    rows = [json.loads(line) for line in Path(
        f"/workspace/results/reference-{split}.jsonl").read_text().splitlines()]
    assert len(rows) == 16 and all(row["reward"] == 1 for row in rows), split
print("Both reference splits passed")
PY
```

**每个 reference 结果都须为
reward=1**，题数各16；这条命令的退出码0本身不能证明全部通过。参考补丁只用于验收环境，不进入模型请求。若环境不通过，先诊断，不能依据模型表现重新选题。

第二台机器复制 `tasks/dapo、tasks/swe-train、tasks/swe-eval`（包含 manifest），按各 SWE 清单运行
`pull-images` 并重新验收 reference。镜像冻结为 registry digest；记录 Python 镜像的 digest并在第二台使用相同
digest。SWE 正式镜像／清单目前尚未实测。

## 2．每次选择一种组合

下面以 `cc-swe` 为例；依次换成表中的其他 CASE，**逐组运行**。每组训练均从同一初始模型重新开始，使用新 trial 和结果目录。

```bash
source /workspace/2x2.env
export CASE=cc-swe
case "$CASE" in
  hermes-dapo|cc-dapo)
    export BENCHMARK=dapo_math CONFIG=dapo.yaml
    export TRAIN_TASKS=/workspace/tasks/dapo/train.jsonl
    export EVAL_TASKS=/workspace/tasks/dapo/eval.jsonl
    export EVAL_LIMIT=200 CONCURRENCY=32 ;;
  hermes-swe|cc-swe)
    export BENCHMARK=swe_rebench CONFIG=swe.yaml
    export TRAIN_TASKS=/workspace/tasks/swe-train/tasks.jsonl
    export EVAL_TASKS=/workspace/tasks/swe-eval/tasks.jsonl
    export EVAL_LIMIT=16 CONCURRENCY=4 ;;
  *) exit 1 ;;
esac
case "$CASE" in
  hermes-*) export AGENT_CLASS=examples.task_workflows.service.TaskAgent ;;
  cc-*) export AGENT_CLASS=examples.task_workflows.service.CCTaskAgent ;;
esac
export TRIAL="$TASK_HW-$CASE-$(date -u +%Y%m%dT%H%M%SZ)"
export CASE_ROOT="/workspace/results/$TRIAL"
mkdir "$CASE_ROOT" || exit 1
export AREAL_HOME="$CASE_ROOT/areal-home"
export HERMES_HOME="$CASE_ROOT/hermes-home"
export AREAL_TASK_TOOL_ROOT="$CASE_ROOT/tools"
mkdir -p "$HERMES_HOME"
printf 'model:\n  context_length: 65536\n' > "$HERMES_HOME/config.yaml"
unset HERMES_MAX_TURNS AREAL_CC_MAX_TURNS
export AGENT_SERVICE="$TRIAL-agent"
export INF_SERVICE="$TRIAL-eval"
env | sed -n '/^\(CASE\|TRIAL\|CASE_ROOT\|BENCHMARK\|CONFIG\|TRAIN_TASKS\|EVAL_TASKS\|EVAL_LIMIT\|CONCURRENCY\|AGENT_CLASS\|AGENT_SERVICE\|INF_SERVICE\|AREAL_HOME\|HERMES_HOME\|AREAL_TASK_TOOL_ROOT\)=/p' \
  | sed 's/^/export /' > "$CASE_ROOT/case.env"
```

工作路径不要含空格。其他终端需先 source 公共环境，再 source 本组 `case.env`。Hermes 恢复原生默认90轮；CC
无额外轮数上限，原生循环和系统提示与 Hermes 不同。一次 episode 是一道题；1 pair 可并发处理多题，客户端 concurrency 控制在途 episode
数，工具／评分也可能占据该名额。

## 3．八卡推理评测

所有组合均经过 v2 网关：Hermes 使用 OpenAI 接口，CC 使用 `/v1/messages`。不要同时启动旧的裸 SGLang 八卡服务。

```bash
export AREAL_ROLLOUT_TRACE_DIR="$CASE_ROOT/eval-rollout-batches"
export AREAL_INFERENCE_URL=http://127.0.0.1:18080
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 areal inf run \
  --service "$INF_SERVICE" --detach --scheduler local --port 18080 \
  --admin-api-key "$AREAL_INFERENCE_ADMIN_KEY" \
  --model task-model --backend sglang:d8 \
  --model-path "$AREAL_MODEL_PATH" --tokenizer-path "$AREAL_MODEL_PATH" \
  --engine-args "--context-length 65536 --mem-fraction-static 0.5 --dtype bfloat16" \
  --proxy-args "--tool-call-parser qwen --reasoning-parser '' --engine-max-tokens 65536 --request-timeout 1800"
areal inf status --service "$INF_SERVICE"
curl -fsS -H "Authorization: Bearer $AREAL_INFERENCE_ADMIN_KEY" \
  "$AREAL_INFERENCE_URL/v1/models"

CUDA_VISIBLE_DEVICES="" areal agent run --service "$AGENT_SERVICE" \
  --agent "$AGENT_CLASS" --num-pairs 1 --session-timeout 7200 \
  --admin-api-key "$AREAL_AGENT_ADMIN_KEY"
areal agent status --service "$AGENT_SERVICE"
export AREAL_AGENT_URL="$(areal agent status --service "$AGENT_SERVICE" --json \
  | python -c 'import json,sys; s=json.load(sys.stdin); assert s["running"],s; print(s["gateway_url"])')"

python -m examples.task_workflows.client --benchmark "$BENCHMARK" \
  --tasks "$EVAL_TASKS" --output "$CASE_ROOT/eval-smoke" \
  --agent-url "$AREAL_AGENT_URL" --inference-url "$AREAL_INFERENCE_URL" \
  --model task-model --mode eval --limit 4 --concurrency 4 --request-timeout 7200
```

所有组件健康，四题 completed=4、infrastructure_errors=0、not_submitted=0后，再运行：

```bash
python -m examples.task_workflows.client --benchmark "$BENCHMARK" \
  --tasks "$EVAL_TASKS" --output "$CASE_ROOT/eval-baseline" \
  --agent-url "$AREAL_AGENT_URL" --inference-url "$AREAL_INFERENCE_URL" \
  --model task-model --mode eval --limit "$EVAL_LIMIT" \
  --concurrency "$CONCURRENCY" --request-timeout 7200
cat "$CASE_ROOT/eval-baseline/summary.json"
areal inf stop --service "$INF_SERVICE" --grace 10
```

`sglang:d8` 由 v2 启动8个TP1 worker，各自对应 DataProxy；**不要再传 --dp-size
8**。启动逐副本进行，首次需要数分钟。`--detach` 在服务准备完成后返回；Agent Service
也在后台运行。DAPO32／SWE4并发是起点，不是硬件实测最优值；比较时记录实际设置。

## 4．4卡 Actor＋4卡 Rollout 常驻训练

先确认上节评测服务已停止，八卡没有其他作业。本节先让四种组合分别完成 **16题、batch4、4次更新**。它只验收闭环，不能证明正式长序列训练显存安全或学习有效。

### 终端一：Trainer

新终端先执行下列两条，将路径中的 TRIAL 换成本组的实际目录名；终端二也执行：

```bash
source /workspace/2x2.env
source /workspace/results/TRIAL/case.env
```

下面共用配置，保持原 PPO 奖励和优化语义，仅设置本次规模、布局、学习率和记录开关：

```bash
export AREAL_TASK_RUN_ROOT="$CASE_ROOT/train"
export AREAL_TASK_STATE_ROOT="$AREAL_TASK_RUN_ROOT/state"
export AREAL_ACTOR_ADMIN_KEY="$AREAL_INFERENCE_ADMIN_KEY"
export AREAL_ACTOR_LIFECYCLE=resident
export AREAL_CHECKPOINT_ESTIMATED_BYTES=0
export AREAL_ROLLOUT_TRACE_DIR="$AREAL_TASK_RUN_ROOT/rollout-batches"
export N=16 EPOCHS=1 BATCH=4 SAVE_EVERY=4
export UPDATES=$((N * EPOCHS / BATCH))
test $((N % BATCH)) -eq 0 || exit 1
mkdir "$AREAL_TASK_RUN_ROOT" || exit 1
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false \
python -m examples.task_workflows.train \
  --config "examples/task_workflows/configs/$CONFIG" \
  experiment_name=task-workflows trial_name="$TRIAL" \
  total_train_epochs="$EPOCHS" total_train_steps="$UPDATES" \
  train_dataset.batch_size="$BATCH" actor.optimizer.lr=1e-6 \
  rollout.backend=sglang:d4 rollout.max_concurrent_rollouts=32 \
  'rollout.agent.reasoning_parser=""' +rollout.agent.engine_max_tokens=65536 \
  +sglang.dp_size=1 sglang.mem_fraction_static=0.5 \
  'actor_lifecycle.rollout_devices=[4,5,6,7]' \
  saver.freq_steps="$SAVE_EVERY" \
  perf_tracer.enabled=true +perf_tracer.save_interval=1 \
  perf_tracer.session_tracer.enabled=true \
  2>&1 | tee "$AREAL_TASK_RUN_ROOT/trainer.log"
trainer_rc=$?
printf '%s\n' "$trainer_rc" > "$AREAL_TASK_RUN_ROOT/trainer.exit"
```

该终端持续运行，等待
`RolloutControllerV2 ... Gateway: http://...`。训练启动自己的推理网关，**不能使用评测端口18080**。不要另外启动
`areal inf`；Agent Service 可继续使用。

### 终端二：提交训练任务

先 source 公共环境和本组 case.env，读取本轮 Trainer 日志中的 Rollout 网关：

```bash
export AREAL_TASK_RUN_ROOT="$CASE_ROOT/train"
export N=16 EPOCHS=1
export AREAL_AGENT_URL="$(areal agent status --service "$AGENT_SERVICE" --json \
  | python -c 'import json,sys; s=json.load(sys.stdin); assert s["running"],s; print(s["gateway_url"])')"
export AREAL_INFERENCE_URL="$(python - <<'PY'
import os, re
from pathlib import Path
text = Path(os.environ["AREAL_TASK_RUN_ROOT"], "trainer.log").read_text()
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
urls = re.findall(r"RolloutControllerV2[^\n]*Gateway: (http://\S+)", text)
assert urls, "Rollout gateway is not ready; inspect trainer.log"
print(urls[-1])
PY
)"
test -n "$AREAL_INFERENCE_URL" || exit 1
set -o pipefail
client_rc=0
for epoch in $(seq 1 "$EPOCHS"); do
  python -m examples.task_workflows.client --benchmark "$BENCHMARK" \
    --tasks "$TRAIN_TASKS" --output "$AREAL_TASK_RUN_ROOT/client-epoch-$epoch" \
    --agent-url "$AREAL_AGENT_URL" --inference-url "$AREAL_INFERENCE_URL" \
    --model "$AREAL_MODEL_PATH" --mode resident --limit "$N" \
    --concurrency "$CONCURRENCY" --request-timeout 7200 \
    2>&1 | tee "$AREAL_TASK_RUN_ROOT/client-epoch-$epoch.log"
  client_rc=$?
  test "$client_rc" -eq 0 || break
done
printf '%s\n' "$client_rc" > "$AREAL_TASK_RUN_ROOT/client.exit"
```

两个终端的 N／EPOCHS 必须一致。每轮复用同一清单顺序，模型／工具重新执行；异步完成顺序和内部请求交错不保证相同。题目数是 episode
数，同题多次模型请求可能形成多条训练序列。

### 验收后扩展训练量

每个新规模都创建新 CASE_ROOT／TRIAL，重新执行公共选择和两端命令；不能重复使用已存在的输出目录。下表是规模示例，不是四组都已经调优：

| 场景             | N／轮 | EPOCHS | BATCH | UPDATES | SAVE_EVERY |
| ---------------- | ----- | ------ | ----- | ------- | ---------- |
| 四组训练冒烟     | 16    | 1      | 4     | 4       | 4          |
| SWE 初步重复训练 | 16    | 5      | 4     | 20      | 4          |
| 原 DAPO 正式目标 | 1000  | 5      | 40    | 125     | 25         |

**原 DAPO batch40 配方仍受 OOM 阻塞。**
H20须先解决原失败规模及64K完整前后向的显存问题；H200也要单独验证。n_mbs只给最少微批数，分配器参数不是显存安全保证；本文未加入未经验证的修复，也未缩短原生轮数或上下文。SWE
的20次更新仅用于试跑，正式规模另定。

成功必须同时满足：每轮 N题全部完成且基础设施错误／未提交为0；Trainer完成预期 UPDATES、权重发布及最后 HF
导出；trainer.exit与client.exit均为0。客户端成功不等于训练成功。HF权重不是含优化器／RNG的可恢复 checkpoint，失败后不能据此宣称精确续训。

Trainer正常退出后，在训练输出的 checkpoints 下找到最终 HF 目录，确认 config、tokenizer、权重／分片索引完整。将下面占位路径换成实际目录：

```bash
export AREAL_MODEL_PATH=/absolute/path/to/final-hf-export
```

重新执行第3节的 `areal inf run`／状态检查部分，加载该目录，采集目录换为
`"$CASE_ROOT/eval-after-rollout-batches"`；复用仍健康的 Agent Service，不重复启动同名 Agent。客户端模型名仍为
`task-model`，完整评测的输出换为 `"$CASE_ROOT/eval-after"`。保持同一 eval清单、采样和工具设置，比较基线与训练后结果。其他组合
source公共环境、重新加载原初始模型训练。

## 5．Trace：现有记录与模拟所缺的信息

| 层次          | 现在可采集                                                                           | 后续 simulation 还需补齐                                           |
| ------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Episode／调用 | simulation_trace的题内调用ID／依赖／工具池与服务耗时，task_index／available_at       | 环境创建／清理／评分的独立span；并发模型分支仍不支持               |
| Rollout 请求  | 逐次响应usage的实际输入／输出长度、completion ID／结束原因，缺失保持null             | 缓存／有效prefill量、引擎入队／首末token／attempt和版本段          |
| Rollout batch | prefill／decode／mixed请求数、token汇总、CUDA event区间                              | 参与请求ID与长度形状、请求到batch关联、KV块及抢占／重算事件        |
| Actor／控制   | perf_tracer阶段；exported-workloads记录成功导出的真实序列长度及attention／loss token | step／rank／microbatch、packing／padding、实际消费与发布的完整关联 |

上述命令启用了已有 batch采集和 Trainer
perf_tracer；逐调用墙钟记录自动保存。SGLang采集按普通TP／DP实现，GPU兼容性和开销待实测；见[字段与时间口径](../../docs/zh/best_practices/perf_profiling.md#sglang-batch-%E8%AF%B7%E6%B1%82%E6%95%B0%E4%B8%8E-token-%E6%97%B6%E5%BA%8F)。不能把预热流量计入正式工作量，也不能把
mixed 的时长重复相加。

采集粒度采用“**调用级依赖＋引擎每batch摘要**”，不全程采每请求每token的内容／时间或逐kernel
profiler。长度来自引擎实际tokenization，不从字符数估计；生成长度是实际值，不用max_tokens代替。后续事件还需统一schema、run／episode／call／attempt身份、主机／进程单调时钟和墙钟校准，保留跨主机对齐误差与丢失事件标记。

逐调用长度／依赖和导出工作量已补采，见[字段与限制](README.md#%E8%AF%84%E5%88%86%E4%B8%8E%E8%AE%B0%E5%BD%95)。代码同步后重启Agent与推理／训练进程才对新运行生效；26项离线正常路径检查通过，真实GPU待验收，见[检查记录](../../../artifacts/expert-trace-collection-20261023/offline-validation.json)。现有调用墙钟不拆成prefill／decode；v2会等底层生成完成再构造流式响应，SDK首包不能代表真实首token。

对[当前工作负载输入](../../areal/elastic/contracts.py)，应区分：

- **模拟输入**：模型调用长度／依赖、工具调用耗时／资源池已写入simulation_trace；客户端清单就绪时间与导出工作量分别落盘。先检查lengths_complete／dependencies_complete；缓存未知不能擅自填0。episode后处理、实际消费／初态及目标配置仍须组装，导出不代表已训练。
- **校准／验证记录**：逐batch执行时间、成员、query／已有context及KV状态。Expert自行组batch并推进decode，历史成员列表和逐token时间不是运行它的必需输入；但当前聚合trace不足以完整核验mixed／chunked服务成本。采集器的context_tokens包含完整prompt，Expert的同名字段指当前query之前的已缓存上下文，必须转换口径。

现有调度为decode-first＋完整长度KV预留的显式假设，不是SGLang调度复刻。Catalog模型加载器目前只覆盖qwen3_moe，内置Qwen3-30B-A3B；1.7B
dense适配及H20／H200目标校准仍待完成，不能套用旧系数。Reference源码与冻结校准未改，见[核查记录](../../../artifacts/expert-trace-audit-20261023/audit.json)。

先用已有数据做运行时间线和描述统计；模拟新配置时根据依赖重新产生下游调用到达时间，重新计算排队／服务时间，不能把历史排队时间固定当计算成本。固定工作负载模拟不能证明改变训练后仍产生相同轨迹或学习质量。

每组保留：代码快照（含未提交改动）、包版本、模型／任务／镜像摘要、解析后配置、客户端和Trainer日志／退出码、episodes／summary、HF导出及两类trace。解析后的配置和CLI状态可能含密钥，分享前脱敏。当前trace缺口不能在完成后凭图补造。

## 6．进度、异常和收尾

客户端有逐题进度条，结束看 summary.json；训练同时检查实际更新数、发布和HF目录。日志、trace均在 CASE_ROOT及train子目录；额外终端可运行：

```bash
nvidia-smi
df -h /workspace
areal agent status --service "$AGENT_SERVICE"
areal agent logs --service "$AGENT_SERVICE" --component worker-0 --lines 80
```

评测期间另用 `areal inf status／logs --service "$INF_SERVICE"`；训练期间查看本轮 trainer.log。出现
OOM、轨迹提交错误或客户端提前退出时，保留现场；仅对本轮前台Trainer按Ctrl+C并检查子进程释放，不能用全局pkill，也不要自动重试生成填满训练batch。

全部结束后，只停止本组专用服务，再退出容器：

```bash
areal agent stop --service "$AGENT_SERVICE" --grace-period 10
areal inf stop --service "$INF_SERVICE" --grace 10
exit
```

回到宿主机后：

```bash
docker exec "$TASK_CONTAINER" chown -R "$(id -u):$(id -g)" /workspace/results
docker stop -t 30 "$TASK_CONTAINER"
```

停止容器前确认本轮Trainer已退出，不影响其他任务；结果保留在 TASK_HOST_ROOT/results。每组串行测试时可保留容器继续下一组，所有组结束后再停容器。
