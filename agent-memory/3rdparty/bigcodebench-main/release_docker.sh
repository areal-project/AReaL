# argument version

set -eux

while getopts "v:" opt; do
  case $opt in
    v)
      version=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

if [ -z "$version" ]; then
  echo "version is required"
  exit 1
fi

export PYTHONPATH=$PWD pytest tests

docker buildx create --name multiplatform-builder --use || true
docker buildx use multiplatform-builder

# Build and push evaluate image
docker buildx build --platform linux/amd64 \
    -f Docker/Evaluate.Dockerfile . \
    -t bigcodebench/bigcodebench-evaluate:$version \
    -t bigcodebench/bigcodebench-evaluate:latest \
    --push

# Build and push gradio image
docker buildx build --platform linux/amd64 \
    -f Docker/Gradio.Dockerfile . \
    -t bigcodebench/bigcodebench-gradio:$version \
    -t bigcodebench/bigcodebench-gradio:latest \
    --push