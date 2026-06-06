# Add Prompt Template

1. Add .yaml template of new encoding to src/prompts/templates
2. Revise scripts/run_aggregation_experiment.py
    ```
    pairwise_template = prompt_manager.get_template("aggregation_pairwise_comparison")
    ``` 
    to 
    ```
    pairwise_template = prompt_manager.get_template("NAME_IN_YOUR_YAML")
    ```

    You can get solution 1 and solution 2 by using `{solution1}` and `{solution2}` in .yaml file.
# Run Experiments
### 1. Open Apptainer
```
workdir=YOUR_DIR/LLM_Test-time_Scaling
srun --mpi=pmi2 --job-name yl-100mattn --ntasks=1 --gres=gpu:8 --chdir=$workdir \
    --cpus-per-task=100 --mem=1500G --pty \
    singularity shell --nv --no-home --writable-tmpfs \
    --bind /storage:/storage /storage/openpsi/images/areal-latest.sif
```

### 2. (Start Service) 
1. Use `hostname -I` to retrieve the current node's IP address, copy the one starting with `10.18`, and then start the service.
2. To run the new-encoding service, `source activate` the corresponding environment.
3. Command.
Start the standard Qwen3-4B service:
    ```
    vllm serve /storage/openpsi/models/Qwen__Qwen3-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --served-model-name qwen3-4b
    ```
    Start the new-encoding service:
    ```
    python -m vllm.entrypoints.openai.api_server \
    --model /storage/openpsi/models/Qwen__Qwen3-4B \
    --served-model-name qwen3-4b-newenc \
    --hf-overrides '{"rope_parameters": {"rope_type": "chunked"}}' \
    --tensor-parallel-size 8
    ```
    **Ensure you add --served-model-name qwen3-4b-newenc.**

### 2. (Run Experiment)
1. Set up running environment:
    ```
    export PYTHONPATH=$(pwd)
    pip install litellm
    export OPENAI_API_BASE=http://<YOUR_IP_ADDRESS>:8000/v1
    export OPENAI_API_KEY=dummy
    ```
2. Generate the result file for qwen3-4b.

    **Direct generation (no turn-wise scaling)** — generates N solutions in parallel without reflection/refinement:
    ```
    python scripts/run_imobench_experiment.py --model qwen3-4b
    ```
    By default this runs `direct_generation` with `n_samples=8`. To switch back to turn-wise scaling, edit the `experiments` list in the script and uncomment the desired strategy (e.g., `no_feedback_sequential_2*8`).

    Note: For dataset formatting, please refer to `scripts/download_benchmark_data.py`.
3. Pairwise Aggregation.
After generating the results, perform pairwise aggregation using the following command:
    ```
    python -m scripts.run_aggregation_experiment \
            --result-files /path/to/your/generated/results \
            --output-dir aggregation_experiments \
            --model_name openai/qwen3-4b(default) or openai/qwen3-4b-newenc
    ```
    
## Latest encoding
### Start server
1. Set up the env, use env in zzy's directory
```
source /storage/openpsi/users/zzy/.zzy-enc-2/bin/activate
```
2. Start vllm server
```
cd /storage/openpsi/users/zzy/zzy-encoding-2 ## Important

python -m vllm.entrypoints.openai.api_server     --model /storage/openpsi/models/zzy/Qwen__Qwen3-4B/     --tokenizer /storage/openpsi/models/zzy/Qwen__Qwen3-4B/     --hf-overrides '{"architectures": ["Qwen3ChunkedForCausalLM"], "chunk_start_token_id": 151669, "chunk_end_token_id": 151670}'     --trust-remote-code --tensor-parallel-size 8 --served-model-name qwen3-4b & ## Be sure to use model in my path, as it adds new tokens
```

3. Run aggregate experiment
3.1.change scripts/run_aggregation_experiment.py, line 569's "aggregation_pairwise_comparison" to "aggregation_pairwise_new_encoding_comparison"

3.2 Deactivate the env
```
deactivate
```
3.3 Then run as normal
```
    python -m scripts.run_aggregation_experiment \
            --result-files /path/to/your/generated/results \
            --output-dir aggregation_experiments \
            --model_name openai/qwen3-4b
```