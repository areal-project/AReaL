# 单机 CPU sandbox 部署

来自现有 H20 的准备配方，**本轮未在空白环境重新构建验收**。先按 [ENVIRONMENT.md](../../ENVIRONMENT.md) 检出固定
AEnvironment／SWEAgent 源码。此步骤只部署 CPU sandbox，不启动训练。

Linux 主机需 Docker、Helm 3、kind、kubectl、Go、Python 3。现有配方使用 kind v0.33.0、kind node／kubectl
v1.34.11、Go 1.26.9。以下适用于独立开发集群；Redis 注册信息不持久化，重建后需重新注册。

## 服务构建

在 GPU 主机执行，`DEMO_HOME` 与启动训练容器时的宿主目录相同；源码若在容器内下载，宿主可从其 bind mount 访问。

```bash
export DEMO_HOME=/path/to/elastic-demo-data
export SETUP="$PWD/examples/elastic/swe/deploy"
export AENV_ROOT="$DEMO_HOME/src/AEnvironment"
export SWE_AGENT_ROOT="$DEMO_HOME/src/AReaL-SWEAgent"
export BUILD="$DEMO_HOME/build"
export KUBECONFIG="$DEMO_HOME/private/kubeconfig"
mkdir -p "$BUILD/bin/swe_eval" "$DEMO_HOME/private"

(cd "$SWE_AGENT_ROOT/cc_golang_bash" && CGO_ENABLED=0 go build -o "$BUILD/bin/persistent-bash" . && go test ./...)
(cd "$AENV_ROOT/controller" && CGO_ENABLED=0 go build -o "$BUILD/bin/controller" ./cmd)
(cd "$AENV_ROOT/envhub" && CGO_ENABLED=0 go build -o "$BUILD/bin/envhub" .)
(cd "$AENV_ROOT/api-service" && CGO_ENABLED=0 go build -o "$BUILD/bin/api-service" .)
cp "$AENV_ROOT"/aenv/builtin-envs/swebench/src/eval/*.py "$BUILD/bin/swe_eval/"

docker pull busybox:1.37.0
docker pull redis@sha256:48501c5ad00d5563bc30c075c7bcef41d7d98de3e9a1e6c752068c66f0a8463b
docker tag redis@sha256:48501c5ad00d5563bc30c075c7bcef41d7d98de3e9a1e6c752068c66f0a8463b redis:alpine
docker pull kindest/node:v1.34.11
docker build -f "$SETUP/Dockerfile.shell" -t autopilot-swe-shell:0.1.0 "$BUILD/bin"
for component in controller envhub api-service; do
  docker build -f "$SETUP/Dockerfile.service" --build-arg COMPONENT="$component" \
    -t "autopilot-aenv-$component:0.1.0" "$BUILD/bin"
done

kind create cluster --name autopilot-swe --image kindest/node:v1.34.11 \
  --config "$SETUP/kind.yaml" --kubeconfig "$KUBECONFIG" --wait 120s
kind load docker-image --name autopilot-swe \
  autopilot-swe-shell:0.1.0 autopilot-aenv-controller:0.1.0 \
  autopilot-aenv-envhub:0.1.0 autopilot-aenv-api-service:0.1.0 redis:alpine
```

服务镜像在本地构建，不是公开 registry 上已有同名镜像。基础 BusyBox、Redis、kind image 目前仍使用 tag，未完成 digest
固定；需在干净环境验收时记录解析出的 digest。八个任务镜像的 digest 已固定。

## 任务镜像与注册

在仓库根目录执行。八个压缩镜像合计约 22 GB，Docker 与 kind 会额外占用存储；这不包含 checkpoint 空间。

```bash
python3 - <<'PY'
import json
import subprocess
images = json.load(open('examples/elastic/reproduction/task-images.json'))
for row in images:
    pinned = row['image'].rsplit(':', 1)[0] + '@' + row['digest']
    alias = 'autopilot-swe-tasks:' + row['task'].lower()
    subprocess.run(['docker', 'pull', pinned], check=True)
    subprocess.run(['docker', 'tag', pinned, alias], check=True)
    subprocess.run(['kind', 'load', 'docker-image', '--name', 'autopilot-swe', alias], check=True)
PY

helm dependency update "$AENV_ROOT/deploy" --skip-refresh
helm lint "$AENV_ROOT/deploy" -f "$SETUP/aenv-values.yaml"
helm upgrade --install autopilot-aenv "$AENV_ROOT/deploy" \
  --kubeconfig "$KUBECONFIG" --namespace aenv \
  -f "$SETUP/aenv-values.yaml" --wait --timeout 180s
kubectl --kubeconfig "$KUBECONFIG" get pods -n aenv
```

chart 自己创建命名空间，不加 `--create-namespace`。`aenv-values.yaml` 显式设置 sandbox namespace、禁用重复子
chart。每个任务 pod 有 shell 服务与真实任务容器，任务容器不挂载宿主 Docker socket 或 service-account token。

下列两个转发分别在独立终端保持运行；已有转发时不重复启动：

```bash
kubectl --kubeconfig "$KUBECONFIG" -n aenv port-forward --address 127.0.0.1 \
  svc/api-service 8080:8080 8081:8081
kubectl --kubeconfig "$KUBECONFIG" -n aenv port-forward --address 127.0.0.1 \
  svc/envhub 8083:8083
```

注册一次，后续只查询：

```bash
curl --fail-with-body -H 'Content-Type: application/json' \
  --data-binary @"$SETUP/environment.json" http://127.0.0.1:8083/env/
curl --fail-with-body http://127.0.0.1:8083/env/autopilot-swe-bash/0.1.0
```

训练容器使用 host network 才能按这些 loopback 地址访问。容器内必须能执行 kubectl，并使用挂载后的 kubeconfig 路径。检查任务节点没有
DiskPressure，再回到环境说明完成 runtime 配置及八题空／gold 验证。
