# Source inside the configured container after activating the isolated Python.
# Its inherited Python packages are outside sys.prefix, so expose the existing
# shared libraries explicitly instead of reinstalling CUDA/cuDNN.
export CUDNN_PATH
CUDNN_PATH=$(python3 -c 'from importlib.metadata import distribution; print(distribution("nvidia-cudnn-cu12").locate_file("nvidia/cudnn"))')
export LD_LIBRARY_PATH="$CUDNN_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
