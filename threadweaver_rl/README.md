# ThreadWeaver RL

ThreadWeaver has a reinforcement learning component for training large language models to generate parallel reasoning paths when solving complex mathematical problems. Built on top of [VERL (Volcano Engine Reinforcement Learning)](https://github.com/volcengine/verl) ([source commit](https://github.com/verl-project/verl/tree/0d4541f397828843525b3f3a7eadff03d56ff24c)), ThreadWeaver enables models to explore multiple solution strategies simultaneously, improving both solution quality and computational efficiency.

## Overview

ThreadWeaver trains language models to generate structured reasoning with special tokens that enable parallel exploration of solution paths:

- **`<Parallel>...</Parallel>`**: Marks a block where multiple reasoning threads can execute in parallel
- **`<Thread>...</Thread>`**: Denotes individual reasoning threads within a parallel block
- **`<think>...</think>`**: Wraps the model's chain-of-thought reasoning
- **`<Conclusion>...</Conclusion>`**: Contains the final answer

This parallel reasoning approach allows models to:
- Explore multiple solution strategies simultaneously
- Improve solution robustness through diverse reasoning paths
- Accelerate training and inference through parallel computation
- Achieve better mathematical problem-solving performance

### Reward Function

The reward function computes rewards based on three components:

1. **Correctness Reward** (`r_correct`):
   - +1.0 for correct answers
   - 0.0 for incorrect answers
   - Uses symbolic math verification and specialized graders

2. **Acceleration Ratio Reward** (`r_accel`):
   - Measures efficiency of parallel reasoning
   - Formula: `r_accel = factor * min(acceleration_ratio - 1.0, clip_max)`
   - Where acceleration ratio = sequential cost / parallel cost
   - Default factor: 0.5, clip_max: 0.2

3. **Combined Reward**:
   ```
   reward = r_correct + r_accel
   ```

The reward function also tracks detailed statistics about parallel structure, thread counts, and token usage.

## Installation
Please follow the veRL installation instructions at https://github.com/volcengine/verl.

We provide versions of packages:
```bash
pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.3 sympy==1.13.1 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 torchao==0.11.0 transformers==4.51.1 datasets==3.6.0 tokenizers==0.21.1 huggingface-hub==0.31.4 safetensors==0.5.3 compressed-tensors==0.9.3 openai==1.75.0 sglang==0.4.6.post1 sgl-kernel==0.1.0 xgrammar==0.1.18 trl==0.19.0 accelerate==1.7.0 peft==0.15.2 deepspeed==0.17.0 liger_kernel==0.5.10 xformers==0.0.29.post2 wandb==0.21.0 tensorboard==2.19.0 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cudnn-cu12==9.1.0.70 nvidia-nccl-cu12==2.21.5 nvidia-cublas-cu12==12.4.5.8 nvidia-cufft-cu12==11.2.1.3 nvidia-curand-cu12==10.3.5.147 nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 nvidia-nvtx-cu12==12.4.127 nvidia-nvjitlink-cu12==12.4.127 tqdm==4.67.1 termcolor==3.1.0 packaging==25.0 typing-extensions==4.13.2 pyyaml==6.0.2 regex==2024.11.6 psutil==7.0.0 filelock==3.18.0 requests==2.32.3 pyzmq==26.4.0 orjson==3.10.18 partial-json-parser==0.2.1.1.post5 flask==2.3.3 fastapi==0.115.12 uvicorn==0.34.2 uvloop==0.21.0 python-multipart==0.0.20 pylatexenc==2.10 codetiming dill hydra-core pybind11 pre-commit "ray[default]==2.52.1" torchdata "pyarrow>=19.0.0" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" latex2sympy2_extended math_verify gunicorn==23.0.0 vllm==0.8.5.post1
pip install flash-attn==2.7.4.post1 opentelemetry-sdk==1.39.1 opentelemetry-exporter-prometheus==0.60b1 --no-build-isolation
```

### Prepare Your SFT Model

ThreadWeaver requires a supervised fine-tuned (SFT) model as initialization. The model should be trained to:
- Use `<think>...</think>` tokens for reasoning
- Generate answers in `\boxed{}` format
- Use the parallel reasoning structure as specified in the ThreadWeaver paper.

See the [ThreadWeaver SFT](../threadweaver_sft/README.md) documentation for SFT training instructions.

Both the SFT and RL can be run on a single node with 8x80G A100 or H100 GPUs. Multi-node training is also supported.

### Single-Node Training
```bash
# ThreadWeaver
export VLLM_USE_V1=1
MODEL_PATH=../threadweaver_sft/ckpts/Q3-8B-131072-SFT
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files="../threadweaver_sft/data/mult-10k-par_pq/train.parquet" data.val_files="../threadweaver_sft/data/mult-10k-par_pq/val.parquet" data.filter_overlong_prompts=True data.train_batch_size=128 data.val_batch_size=512 data.max_prompt_length=9216 data.max_response_length=8192 actor_rollout_ref.model.path=$MODEL_PATH actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null actor_rollout_ref.actor.ppo_mini_batch_size=null actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 actor_rollout_ref.rollout.max_num_batched_tokens=10240 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 actor_rollout_ref.actor.grad_clip=1.0 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.n=8 actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=True algorithm.use_kl_in_reward=False algorithm.norm_adv_by_std_in_grpo=False trainer.critic_warmup=0 trainer.logger=['console','tensorboard','wandb'] trainer.project_name='deepscaler' trainer.experiment_name="1n-p1-nonrm-8k-multv5-10k-par-a0.5am0.2_rva2_par_bfl-1217" trainer.val_before_train=False trainer.n_gpus_per_node="8" trainer.nnodes="1" trainer.save_freq=10 trainer.test_freq=10 trainer.default_hdfs_dir=null trainer.total_epochs=30 actor_rollout_ref.rollout.max_model_len=8192 reward_model.config.acceleration_ratio_reward=1.0 reward_model.config.acceleration_ratio_reward_factor=0.5 reward_model.config.acceleration_ratio_clip_max=0.2 reward_model.config.version=v2 reward_model.config.require_think_end=False reward_model.reward_manager_type=reward_manager_with_server actor_rollout_ref.rollout.agent.num_workers=8 actor_rollout_ref.rollout.agent.enable_parallel_branching=True actor_rollout_ref.rollout.agent_return_expanded_sequences=True actor_rollout_ref.rollout.agent.no_conclusion=true algorithm.broadcast_from_last=True reward_model.config.strip_comma_from_answer=True data.return_raw_chat=True actor_rollout_ref.rollout.mode=async trainer.max_actor_ckpt_to_keep=3

# Sequential Baseline
export VLLM_USE_V1=1
MODEL_PATH=../threadweaver_sft/ckpts/Q3-8B-131072-AR-SFT
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files="../threadweaver_sft/data/mult-10k-par_pq/train.parquet" data.val_files="../threadweaver_sft/data/mult-10k-par_pq/val.parquet" data.filter_overlong_prompts=True data.train_batch_size=128 data.val_batch_size=512 data.max_prompt_length=9216 data.max_response_length=8192 actor_rollout_ref.model.path=$MODEL_PATH actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null actor_rollout_ref.actor.ppo_mini_batch_size=null actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 actor_rollout_ref.rollout.max_num_batched_tokens=10240 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 actor_rollout_ref.actor.grad_clip=1.0 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.n=8 actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=True algorithm.use_kl_in_reward=False algorithm.norm_adv_by_std_in_grpo=False trainer.critic_warmup=0 trainer.logger=['console','tensorboard','wandb'] trainer.project_name='deepscaler' trainer.experiment_name="1n-p1-nonrm-8k-multv5-10k-par-ar-a0.5am0.2_rva2_seq-1217" trainer.val_before_train=False trainer.n_gpus_per_node="8" trainer.nnodes="1" trainer.save_freq=10 trainer.test_freq=10 trainer.default_hdfs_dir=null trainer.total_epochs=30 actor_rollout_ref.rollout.max_model_len=8192 reward_model.config.acceleration_ratio_reward=1.0 reward_model.config.acceleration_ratio_reward_factor=0.5 reward_model.config.acceleration_ratio_clip_max=0.2 reward_model.config.version=v2 reward_model.config.require_think_end=False reward_model.reward_manager_type=reward_manager_with_server actor_rollout_ref.rollout.agent.num_workers=8 actor_rollout_ref.rollout.agent.enable_parallel_branching=False actor_rollout_ref.rollout.agent_return_expanded_sequences=True actor_rollout_ref.rollout.agent.no_conclusion=true reward_model.config.strip_comma_from_answer=True data.return_raw_chat=True actor_rollout_ref.rollout.mode=async trainer.max_actor_ckpt_to_keep=3
```

### Multi-Node Training

1. **Edit the SLURM script**:
```bash
vim multinode_run.slurm
```

Update the following variables:
- `SBATCH --partition=`: Your SLURM partition
- `SBATCH --qos=`: Your QoS
- `SBATCH --account=`: Your account
- `MODEL_PATH`: Path to your SFT checkpoint
- `TRAIN_FILES`: Path to training data
- `VAL_FILES`: Path to validation data

2. **Submit the job**:
```bash
sbatch multinode_run.slurm
```

3. **Monitor progress**:
```bash
tail -f slurm-<job-id>.out
# Or view TensorBoard logs
tensorboard --logdir=./tensorboard_log
```

## TensorBoard

ThreadWeaver logs training metrics to TensorBoard:

```bash
tensorboard --logdir=./tensorboard_log
```

Key metrics:
- `reward/mean`: Average reward per epoch
- `reward/std`: Reward standard deviation
- `metrics/correct_rate`: Fraction of correct answers
- `metrics/acceleration_ratio`: Average acceleration
- `metrics/parallel_usage`: Percentage using parallel blocks
- `loss/policy_loss`: Policy gradient loss
- `loss/value_loss`: Value function loss

### Computing acceleration w.r.t. sequential baseline
The acceleration ratio w.r.t. the sequential baseline is computed as follows:
```
acceleration_ratio = the total number of tokens in the sequential baseline / the total number of tokens in the longest path of the parallel baseline (i.e., the critical path in terms of the number of tokens)
```

The former is listed as the `total_num_tokens` and the latter as the `num_tokens_in_the_longest_thread`

## Reference Results
We train both models for 400 steps. The sequantial model and the parallel model achieve very close accuracy after 400 steps of training.

| Setting | Num Tokens in the Longest Thread | Accuracy (%) |
| :--- | :---: | :--- |
| Sequential Baseline | 3322 | 99.0% |
| Parallel Model (ThreadWeaver) | 2632 | 99.0% |

ThreadWeaver achieves a speedup of 1.26x w.r.t. the sequential baseline in terms of token latency.

![Math Multiplication Performance Comparison](../assets/math_mult_comparison.png)
