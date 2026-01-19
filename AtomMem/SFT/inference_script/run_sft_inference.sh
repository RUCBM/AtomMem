#!/bin/bash
set -x

DATASET_ROOT=./AtomMem/
export RAY_TMPDIR=/nfsdata/huoyupeng/ray_tmp

MODEL_PATH="/nfsdata/models/Qwen3-8B"
VAL_PATH="${DATASET_ROOT}/taskutils/memory_data/musique_multiquery_sft.parquet"
TRAIN_PATH="${DATASET_ROOT}/taskutils/memory_data/musique_multiquery_sft.parquet"
WRITE_PATH="${DATASET_ROOT}/SFT/data/SFT_test.jsonl"
LOG_PATH="${DATASET_ROOT}/log/test.json"
VLLM_PORT=9007

until curl -s "http://localhost:${VLLM_PORT}/v1/models" | grep -q "qwen3-embedding"; do
  ATTEMPT=$((ATTEMPT+1))
  if [ $ATTEMPT -ge 30 ]; then
    echo "[ERROR] vLLM Start Failed"
    kill $VLLM_PID
    exit 1
  fi
  echo Attempt $ATTEMPT to connect to vLLM...
  sleep 10
done

echo The vLLM service is ready. Launching the training process...

python3 -X faulthandler -m verl.trainer.sft_data_inference \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=4000 \
    recurrent.memory.config.max_memorization_length=4096 \
    +recurrent.memory.config.max_long_mem_length=2048 \
    +recurrent.memory.config.max_short_mem_length=2048 \
    recurrent.memory.config.max_chunks=40 \
    recurrent.memory.path='recurrent/impls/memory_long_short_musiqueprompt.py' \
    +inference_api_key="YOUR_API_KEY" \
    +inference_base_url="YOUR_BASE_URL" \
    +inference_model_name="YOUR_MODEL_NAME" \
    +file_path=$WRITE_PATH \
    +log_path=$LOG_PATH \
    +start_index=0 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.train_batch_size=128 \
    data.truncation='center' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    +reward_model.reward_kwargs={"use_llm":True, "base_url": "YOUR_BASE_URL", "api_key": "YOUR_API_KEY", "model_name": "YOUR_MODEL_NAME"} \
    reward_model.reward_manager='thread_agent' \
