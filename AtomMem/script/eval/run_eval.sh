#!/bin/bash
set -euo pipefail

DATASET_ROOT=.

MODEL_PATH=/nfsdata/models/Qwen3-8B

VLLM_PORT=9007

until curl -s "http://localhost:${VLLM_PORT}/v1/models" | grep -q "qwen3-embedding"; do
  ATTEMPT=$((ATTEMPT+1))
  if [ $ATTEMPT -ge 30 ]; then
    echo "[ERROR] vLLM 启动失败"
    kill $VLLM_PID
    exit 1
  fi
  echo "[WAIT] 第 $ATTEMPT 次尝试连接 vLLM 中..."
  sleep 10
done

# ======== hotpotqa ============== 
# ========= hotpotqa 200doc ===========
# fill the api_key and base_url to determine the judge model, in our impletation the judge model is deepseek-V3.1.
python3 -X faulthandler -m script.eval.eval \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=4000 \
    recurrent.memory.config.max_chunks=40 \
    recurrent.memory.config.max_memorization_length=4096 \
    +recurrent.memory.config.max_long_mem_length=2048 \
    +recurrent.memory.config.max_short_mem_length=2048 \
    recurrent.memory.path='recurrent/impls/memory_long_short_multiquery.py' \
    data.train_files=${DATASET_ROOT}/taskutils/memory_data/hotpotqa_dev_multiquery.parquet \
    data.val_files=${DATASET_ROOT}/taskutils/memory_data/hotpotqa_dev_multiquery.parquet \
    data.train_batch_size=200 \
    data.truncation='center' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    reward_model.reward_manager='thread_agent' \
    +log_path=${DATASET_ROOT}/log/hotpotqa_dev.txt \
    +model_name="AtomMem" \
    +base_url="http://localhost:8001/v1/" \
    +reward_model.reward_kwargs.use_llm=true \
    +reward_model.reward_kwargs.base_url="https://api.deepseek.com" \
    +reward_model.reward_kwargs.api_key="" \
    +reward_model.reward_kwargs.model_name="deepseek-chat" \