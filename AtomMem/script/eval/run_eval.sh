#!/bin/bash
set -euo pipefail

DATASET_ROOT=/nfsdata/huoyupeng/Memory_Agent

MODEL_PATH=/nfsdata/huoyupeng/Memory_Agent/taskutils/memory_model/Qwen3-8B_distill_deepseek_v3.2

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
python3 -X faulthandler -m eval.memory_agent.hotpotqa.eval \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=4000 \
    recurrent.memory.config.max_chunks=40 \
    recurrent.memory.config.max_memorization_length=4096 \
    +recurrent.memory.config.max_long_mem_length=2048 \
    +recurrent.memory.config.max_short_mem_length=2048 \
    recurrent.memory.path='recurrent/impls/memory_long_short_multiquery.py' \
    data.train_files=${DATASET_ROOT}/taskutils/memory_data/hotpotqa_multiquery_dev_filtered.parquet \
    data.val_files=${DATASET_ROOT}/taskutils/memory_data/hotpotqa_multiquery_dev_filtered.parquet \
    data.train_batch_size=200 \
    data.truncation='center' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    reward_model.reward_manager='thread_agent' \
    +log_path='/nfsdata/huoyupeng/Memory_Agent/eval/memory_agent/hotpotqa/memory_agent_rl_hotpotqa_200doc_filter.txt' \
    +model_name="memory_agent" \
    +base_url="http://localhost:8001/v1/"