#!/bin/bash
#SBATCH --job-name=mult-seq-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000MB
#SBATCH --partition=partition
#SBATCH --time=3-00:00:00
#SBATCH --account=account
#SBATCH --qos=qos
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=logs/slurm-mult-seq-training-%j.out
#SBATCH --error=logs/slurm-mult-seq-training-%j.out

export VLLM_USE_V1=1
export RAY_memory_monitor_refresh_ms=0
MODEL_PATH=../threadweaver_sft/ckpts/Q3-8B-131072-AR-SFT-20251217_092333
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"

# Getting the node names
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Nodes: ${nodes_array[*]}"
echo "Head: $head_node"

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files="../threadweaver_sft/data/mult-10k-par_pq/train.parquet" data.val_files="../threadweaver_sft/data/mult-10k-par_pq/val.parquet" data.filter_overlong_prompts=True data.train_batch_size=128 data.val_batch_size=512 data.max_prompt_length=9216 data.max_response_length=8192 actor_rollout_ref.model.path=$MODEL_PATH actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null actor_rollout_ref.actor.ppo_mini_batch_size=null actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 actor_rollout_ref.rollout.max_num_batched_tokens=10240 actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 actor_rollout_ref.actor.grad_clip=1.0 actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.n=8 actor_rollout_ref.ref.fsdp_config.param_offload=True actor_rollout_ref.rollout.enforce_eager=False actor_rollout_ref.rollout.free_cache_engine=True algorithm.use_kl_in_reward=False algorithm.norm_adv_by_std_in_grpo=False trainer.critic_warmup=0 trainer.logger=['console','tensorboard'] trainer.project_name='deepscaler' trainer.experiment_name="${SLURM_NNODES}n-p1-nonrm-8k-multv5-10k-par-ar-a0.5am0.2_rva2_seq-1217" trainer.val_before_train=False trainer.n_gpus_per_node="${SLURM_GPUS_PER_NODE}" trainer.nnodes="${SLURM_NNODES}" trainer.save_freq=10 trainer.test_freq=10 trainer.default_hdfs_dir=null trainer.total_epochs=30 actor_rollout_ref.rollout.max_model_len=8192 reward_model.config.acceleration_ratio_reward=1.0 reward_model.config.acceleration_ratio_reward_factor=0.5 reward_model.config.acceleration_ratio_clip_max=0.2 reward_model.config.version=v2 reward_model.config.require_think_end=False reward_model.reward_manager_type=reward_manager_with_server actor_rollout_ref.rollout.agent.num_workers=8 actor_rollout_ref.rollout.agent.enable_parallel_branching=False actor_rollout_ref.rollout.agent_return_expanded_sequences=True actor_rollout_ref.rollout.agent.no_conclusion=true reward_model.config.strip_comma_from_answer=True data.return_raw_chat=True actor_rollout_ref.rollout.mode=async
