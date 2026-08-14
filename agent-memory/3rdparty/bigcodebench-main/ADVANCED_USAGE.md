## 🔥 Advanced Start

To get started, please first set up the environment:

```bash
# If you want to use the evaluate locally, you need to install the requirements in an isolated environment
pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt

# You are strongly recommended to install the bigcodebench dependencies in another environment
pip install bigcodebench --upgrade
```

<details><summary>⏬ Install nightly version <i>:: click to expand ::</i></summary>
<div>

```bash
# Install to use bigcodebench
pip install "git+https://github.com/bigcode-project/bigcodebench.git" --upgrade
```

</div>
</details>

<details><summary>⏬ Using BigCodeBench as a local repo? <i>:: click to expand ::</i></summary>
<div>

```bash
git clone https://github.com/bigcode-project/bigcodebench.git
cd bigcodebench
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Install to use bigcodebench
pip install -e .
```

</div>
</details>

## 🚀 Remote Evaluation

Below are all the arguments for `bigcodebench.evaluate` for the remote evaluation:

#### Required Arguments:
- `--model`: The model to evaluate
- `--split`: The split of the dataset to evaluate
- `--subset`: The subset of the dataset to evaluate

#### Optional Arguments:
- `--root`: The root directory to store the results, default to `bcb_results`
- `--bs`: The batch size, default to `1`
- `--n_samples`: The number of samples, default to `1`
- `--temperature`: The temperature, default to `0.0`
- `--max_new_tokens`: The length of max new tokens, default to `1280`
- `--max_model_len`: The length of max tokens in VLLM, default to `12800`
- `--greedy`: Whether to use greedy decoding, default to `False`
- `--strip_newlines`: Whether to strip newlines, default to `False`, set to `True` to strip newlines for some model series like StarCoder2
- `--direct_completion`: Whether to use direct completion, default to `False`
- `--resume`: Whether to resume the evaluation, default to `True`, set to `False` to re-run the evaluation
- `--id_range`: The range of the tasks to evaluate, default to `None`, e.g. `--id_range 10-20` will evaluate the tasks from 10 to 20
- `--backend`: The backend to use, default to `vllm`
- `--execution`: The execution backend to use, default to `gradio`. You can choose from `e2b`, `gradio`, `local`.
- `--reasoning_effort`: The reasoning effort to use, default to `medium`. You can choose from `easy`, `medium`, `hard` for `o1`, `o3` and `deepseek-reasoner`(soon) models.
- `--base_url`: The base URL of the backend for OpenAI-compatible APIs, default to `None`
- `--instruction_prefix`: The instruction prefix for the Anthropic backend.
- `--response_prefix`: The response prefix for the Anthropic backend.
- `--skip_prefill`: Whether to skip the prefill for vLLM and HF backend, which is useful for reasoning models.
- `--revision`: The revision of the model with the vLLM or HF backend, default to `main`
- `--tp`: The tensor parallel size for the vLLM backend, default to `1`
- `--trust_remote_code`: Whether to trust the remote code, default to `False`
- `--tokenizer_name`: The name of the customized tokenizer, default to `None`
- `--tokenizer_legacy`: Whether to use the legacy tokenizer, default to `False`
- `--samples`: The path to the generated samples file, default to `None`
- `--no_execute`: Whether to not execute the samples, default to `False`
- `--e2b_endpoint`: The API endpoint for remote execution, default to `bigcodebench_evaluator`, you can also use your own E2B API endpoint by cloning the [bigcodebench-evaluator](https://huggingface.co/spaces/bigcode/bigcodebench-evaluator) repo and check `Use via API` at the bottom of the HF space page
- `--gradio_endpoint`: The API endpoint for remote execution, default to `https://bigcode-bigcodebench-evaluator.hf.space/`, you can also use your own Gradio API endpoint by cloning the [bigcodebench-evaluator](https://huggingface.co/spaces/bigcode/bigcodebench-evaluator) repo and check `Use via API` at the bottom of the HF space page
- `--pass_k`: The `k` in `Pass@k`, default to `[1, 5, 10]`, e.g. `--pass_k 1,5,10` will evaluate `Pass@1`, `Pass@5` and `Pass@10`
- `--calibrated`: Whether to use the calibrated samples, default to `True`
- `--save_pass_rate`: Whether to save the pass rate to a file, default to `True`
- `--parallel`: The number of parallel processes, default to `-1`, e.g. `--parallel 10` will evaluate 10 samples in parallel
- `--min_time_limit`: The minimum time limit for the execution, default to `1`, e.g. `--min_time_limit 10` will evaluate the samples with at least 10 seconds
- `--max_as_limit`: The maximum address space limit for the execution, default to `30*1024` (30 GB), e.g. `--max_as_limit 20*1024` will evaluate the samples with at most 20 GB
- `--max_data_limit`: The maximum data segment limit for the execution, default to `30*1024` (30 GB), e.g. `--max_data_limit 20*1024` will evaluate the samples with at most 20 GB
- `--max_stack_limit`: The maximum stack limit for the execution, default to `10`, e.g. `--max_stack_limit 20` will evaluate the samples with at most 20 MB
- `--selective_evaluate`: The subset of the dataset to evaluate, default to `""`. You can pass the index of the tasks to evaluate, e.g. `--selective_evaluate 1,2,3` will evaluate the BigCodeBench/1, BigCodeBench/2 and BigCodeBench/3
- `--check_gt_only`: Whether to only check the ground truths, default to `False`
- `--no_gt`: Whether to not check the ground truths, default to `False`

## 🚀 Full Script

We provide an example script to run the full pipeline for the remote evaluation:

```bash
bash run.sh
```

## 🚀 Local Generation

```bash
# when greedy, there is no need for temperature and n_samples
bigcodebench.generate \
    --model [model_name] \
    --split [complete|instruct] \
    --subset [full|hard] \
    [--greedy] \
    --bs [bs] \
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|openai|mistral|anthropic|google|hf] \
    --tp [TENSOR_PARALLEL_SIZE] \
    [--trust_remote_code] \
    [--base_url [base_url]] \
    [--tokenizer_name [tokenizer_name]]
```

>
The generated code samples will be stored in a file named `[model_name]--bigcodebench-[instruct|complete]--[backend]-[temp]-[n_samples]-sanitized_calibrated.jsonl`. Alternatively, you can use the following command to utilize our pre-built docker images for generating code samples:
>

```bash
# If you are using GPUs
docker run --gpus '"device=$CUDA_VISIBLE_DEVICES"' -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest \
    --model [model_name] \ 
    --split [complete|instruct] \
    --subset [full|hard] \
    [--greedy] \
    --bs [bs] \   
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|openai|mistral|anthropic|google|hf] \
    --tp [TENSOR_PARALLEL_SIZE]

# ...Or if you are using CPUs
docker run -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest \
    --model [model_name] \ 
    --split [complete|instruct] \
    --subset [full|hard] \
    [--greedy] \
    --bs [bs] \   
    --temperature [temp] \
    --n_samples [n_samples] \
    --resume \
    --backend [vllm|hf|openai|mistral|anthropic|google]
```
>
```bash
# If you wish to use gated or private HuggingFace models and datasets
docker run -e HUGGING_FACE_HUB_TOKEN=$token -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments4

# Similarly, to use other backends that require authentication
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments
docker run -e ANTHROPIC_KEY=$ANTHROPIC_KEY -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments
docker run -e MISTRAL_KEY=$MISTRAL_KEY -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments
docker run -e GOOGLE_API_KEY=$OPENAI_API_KEY -v $(pwd):/app -t bigcodebench/bigcodebench-generate:latest # omit other arguments
```
>
Following which, you can run the built container as shown in above.
>
<details><summary>🤔 Structure of `problem`? <i>:: click to expand ::</i></summary>
<div>

* `task_id` is the identifier string for the task
* `entry_point` is the name of the function
* `complete_prompt` is the prompt for BigCodeBench-Complete
* `instruct_prompt` is the prompt for BigCodeBench-Instruct
+ `canonical_solution` is the ground-truth implementation
+ `test` is the `unittest.TestCase` class

</div>
</details>

> [!Note]
>
> **Expected Schema of `[model_name]--bigcodebench-[task]--[backend]-[temp]-[n_samples].jsonl`**
>
> 1. `task_id`: Task ID, which are the keys of `get_bigcodebench()`
> 2. `solution` (optional): Self-contained solution (usually including the prompt)
> 3. `raw_solution` (optional): The raw solution generated by the LLM
>    * Example: `{"task_id": "BigCodeBench/?", "solution": "def f():\n    return 1", "raw_solution": "def f():\n    return 1\nprint(f())"}`


<details><summary>🔎 Checking the compatibility of post-processed code<i>:: click to expand ::</i></summary>
<div>

To double-check the post-processing results, you can use `bigcodebench.syncheck` to check the code validity before and after sanitization, which will print erroneous code snippets and why they are wrong:

```bash
# 💡 If you are storing codes in jsonl:
bigcodebench.syncheck --samples samples.jsonl

# 💡 If you are storing codes in directories:
bigcodebench.syncheck --samples /path/to/vicuna-[??]b_temp_[??]

# 💡 Or change the entrypoint to bigcodebench.syncheck in any pre-built docker image, like 
docker run -it --entrypoint bigcodebench.syncheck -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest --samples samples.jsonl
```

</div>
</details>


## 🚀 Local Evaluation

You are strongly recommended to use a sandbox such as [docker](https://docs.docker.com/get-docker/):

```bash
# Mount the current directory to the container
# If you want to change the RAM address space limit (in MB, 30 GB by default): `--max-as-limit XXX`
# If you want to change the RAM data segment limit (in MB, 30 GB by default): `--max-data-limit`
# If you want to change the RAM stack limit (in MB, 10 MB by default): `--max-stack-limit`
# If you want to increase the execution time limit (in seconds, 240 seconds by default): `--min-time-limit`
docker run -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest --execution local --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl

# If you only want to check the ground truths
docker run -v $(pwd):/app bigcodebench/bigcodebench-evaluate:latest --execution local --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl --check-gt-only
```

...Or if you want to try it locally regardless of the risks ⚠️:

First, install the dependencies for BigCodeBench:

```bash
pip install -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt
```

Then, run the evaluation:

```bash
# ...Or locally ⚠️
bigcodebench.evaluate --execution local --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl
# ...If you really don't want to check the ground truths
bigcodebench.evaluate --execution local --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl --no-gt
# If you want to save the pass rate to a file
bigcodebench.evaluate --execution local --split [complete|instruct] --subset [full|hard] --samples samples-sanitized-calibrated.jsonl --save_pass_rate

# You are strongly recommended to use the following command to clean up the environment after evaluation:
pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); if [ -n \"$pids\" ]; then echo $pids | xargs -r kill; fi;
rm -rf /tmp/*
```

> [!Tip]
>
> If you want to customize the `k` in `Pass@k`, please pass `--pass_k` with a comma-separated string.
> For example, if you want to use `Pass@1` and `Pass@100`, you can pass `--pass_k 1,100`.

> [!Tip]
>
> Do you use a very slow machine?
>
> LLM solutions are regarded as **failed** on timeout (and OOM etc.).
> Specifically, we set the dynamic timeout based on the ground-truth solution's runtime.
>
> Additionally, you are **NOT** encouraged to make your test-bed over stressed while running evaluation.
> For example, using `--parallel 64` on a 4-core machine or doing something else during evaluation are bad ideas...

<details><summary>⌨️ More command-line flags <i>:: click to expand ::</i></summary>
<div>

* `--parallel`: by default half of the cores

</div>
</details>

The output should be like (below is GPT-4 greedy decoding example):

```
Asserting the groundtruth...
Expected outputs computed in 1200.0 seconds
Reading samples...
1140it [00:00, 1901.64it/s]
Evaluating samples...
100%|██████████████████████████████████████████| 1140/1140 [19:53<00:00, 6.75it/s]
BigCodeBench-Instruct-calibrated
Groundtruth pass rate: 1.000
pass@1: 0.568
```

- A cache file named like `samples_eval_results.json` will be cached. Remove it to re-run the evaluation

<details><summary>🤔 How long it would take? <i>:: click to expand ::</i></summary>
<div>

If you do greedy decoding where there is only one sample for each task, the evaluation should take just a few minutes on Intel(R) Xeon(R) Gold 6150 CPU @ 2.70GHz, composed of 2 sockets, with 18 cores per socket. However, if you have multiple samples for each task, the evaluation will take longer.
Here are some tips to speed up the evaluation:

* Use `--parallel $(nproc)`
* Use our pre-evaluated results (see [LLM-generated code](#-LLM-generated-code))

</div>
</details>

## 🔍 Failure Inspection

You can inspect the failed samples by using the following command:

```bash
# Inspect the failed samples and save the results to `inspect/`
bigcodebench.inspect --eval_results sample-sanitized-calibrated_eval_results.json --split complete --subset hard

# Re-run the inspection in place
bigcodebench.inspect --eval_results sample-sanitized-calibrated_eval_results.json --split complete --subset hard --in_place
```

## 📊 Result Analysis

We provide a script to replicate the analysis like Elo Rating and Task Solve Rate, which helps you understand the performance of the models further.

```bash
To run the analysis, you need to put all the `samples_eval_results.json` files in a `results` folder, which is in the same directory as the script.

```bash
cd analysis
python get_results.py
```

## 🐞 Resolved Issues

- [x] Due to [the Hugging Face tokenizer update](https://github.com/huggingface/transformers/pull/31305), some tokenizers may be broken and will degrade the performance of the evaluation. Therefore, we set up with `legacy=False` for the initialization. If you notice the unexpected behaviors, please try `--tokenizer_legacy` during the generation.

- [x] Due to the flakiness in the evaluation, the execution results may vary slightly (~0.2% for Full set, and ~0.6% for Hard set) between runs. We are working on improving the evaluation stability.

- [x] You may get errors like `ImportError: /usr/local/lib/python3.10/site-packages/matplotlib/_c_internal_utils.cpython-310-x86_64-linux-gnu.so: failed to map segment from shared object` when running the evaluation. This is due to the memory limit of the docker container. You can increase the memory limit of the docker container to solve this issue. If the issue persists ,please use the real-time code execution session to evaluate the code in the [leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard).

- [x] We are aware of the issue of some users needing to use a proxy to access the internet. Please use [Remote Evaluation](#-remote-evaluation) to get the accurate results.

## 📜 Citation

```bibtex
@article{zhuo2024bigcodebench,
  title={BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions},
  author={Zhuo, Terry Yue and Vu, Minh Chien and Chim, Jenny and Hu, Han and Yu, Wenhao and Widyasari, Ratnadira and Yusuf, Imam Nur Bani and Zhan, Haolan and He, Junda and Paul, Indraneil and others},
  journal={arXiv preprint arXiv:2406.15877},
  year={2024}
}
```

## 🙏 Acknowledgement

- [EvalPlus](https://github.com/evalplus/evalplus)
