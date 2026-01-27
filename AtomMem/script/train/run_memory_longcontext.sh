#!/bin/bash
set -x

NNODES=1
NGPUS_PER_NODE=8
PROJ_ROOT=.
# enter your swanlab api_key to log the training metrics
export SWANLAB_API_KEY=
# enter your base model path
MODEL_PATH=
VAL_PATH=taskutils/memory_data/hotpotqa_dev_multiquery.parquet
TRAIN_PATH=taskutils/memory_data/hotpotqa_train_multiquery.parquet
EXP=test_rl_training
PROJ_DIR=${PROJ_ROOT}/${EXP}
VLLM_PORT=9007

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=16384
MAX_NEW_TOKEN=4096
PYTHONUNBUFFERED=1

# vllm serve /nfsdata/models/Qwen3-0.6B \
#     --served_model_name=qwen3-embedding \
#     --task embed \
#     --port $VLLM_PORT &

# you must start the embedding service before start training
until curl -s "http://localhost:${VLLM_PORT}/v1/models" | grep -q "qwen3-embedding"; do
  ATTEMPT=$((ATTEMPT+1))
  if [ $ATTEMPT -ge 30 ]; then
    echo "[ERROR] Failed to start embedding service."
    kill $VLLM_PID
    exit 1
  fi
  echo "[WAIT] Attempt ${ATTEMPT}: waiting for embedding service to become available..."
  sleep 10
done

echo "embedding service is ready. Launching the training job."

python3 -X faulthandler -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=4000 \
    recurrent.memory.config.max_chunks=40 \
    recurrent.memory.config.max_memorization_length=4096 \
    +recurrent.memory.config.max_long_mem_length=2048 \
    +recurrent.memory.config.max_short_mem_length=2048 \
    recurrent.memory.path='recurrent/impls/memory_long_short_multiquery.py' \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=10 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=True \
    data.filter_overlong_prompts=False \
    data.train_batch_size=8 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread_agent' \
    +reward_model.use_custom_reward=True \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.99 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='memory_agent_multiquery' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=5