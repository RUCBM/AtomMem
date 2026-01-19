#!/bin/bash

WORK_DIR="./AtomMem/SFT"
GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWANLAB_PROJECT=memory_agent_SFT
export SWANLAB_EXP_NAME=hotpotqa

# 运行 Python 脚本
accelerate launch --config_file=${WORK_DIR}/config/deepspeed_zero3.yaml ${WORK_DIR}/train_script/sft_qwen.py \
    --dataset_name "${WORK_DIR}/SFT/data/SFT_dataset.jsonl" \
    --model_name_or_path "YOUR ORIGINAL MODEL PATH" \
    --output_dir "${WORK_DIR}/taskutils/memory_model/SFT_Qwen3-8B" \
    --max_seq_length 18384 \
    --per_device_train_batch_size 2 \
    --save_steps 1000 \
    --dataloader_drop_last \
    --logging_steps 2 \
    --do_train \
    --gradient_checkpointing_kwargs "$GRADIENT_CHECKPOINTING_KWARGS" \
    --fp16 \
    --num_train_epochs 3 \
    --attn_implementation "flash_attention_2" \
    --torch_dtype bfloat16 \
    --gradient_checkpointing 1 \