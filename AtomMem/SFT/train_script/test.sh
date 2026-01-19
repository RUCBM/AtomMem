echo "ok"
WORK_DIR="my_codebase/PersonaBench/model_training"

python ${WORK_DIR}/script/zero_to_fp32.py ${WORK_DIR}/sft_models/Qwen_mutistep_Qwen_trained ${WORK_DIR}/sft_models/Qwen_mutistep_Qwen_trained \
--safe_serialization