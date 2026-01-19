import json
from transformers import AutoTokenizer
from trl import apply_chat_template

# dataset_dir = "./my_codebase/PersonaBench/model_training/data/gpt_format/mutistep_gpt-4o_train_dataset.jsonl"
# dataset = []
# save_dir = "my_codebase/PersonaBench/model_training/data/llama_train/train_mutistep_4o.jsonl"
# tokenizer = AutoTokenizer.from_pretrained("/yinxr/workhome/zzhong/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct")

# with open(dataset_dir,encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
#         history = data["dialogue_history"]
#         gpt_history = []
#         for dialog in history:
#             new_dialog = [{"role": "user","content":dialog[0]},
#                           {"role":"assistant","content":dialog[1]}]
#             gpt_history.extend(new_dialog)
#         query = data["query"]
#         gpt_history.append({"role":"user","content":query})
#         reformated_data = {
#             "prompt": gpt_history,
#             "completion": [{"role":"assistant","content":data["response"]}]
#                         }
#         reformated_data = apply_chat_template(reformated_data,tokenizer)
#         with open(save_dir,"a+",encoding="utf-8") as g:
#             json.dump(reformated_data,g,separators=(',',':'),ensure_ascii=False)
#             g.write('\n')

dataset_dir = "/home/test/test03/huoyupeng/self-adaptation_LM_reproduce/dataset/SFT_Qwen2.5-72B-restate/filtered_data.jsonl"
dataset = []
save_dir = "/home/test/test03/huoyupeng/self-adaptation_LM_reproduce/dataset/SFT_Qwen2.5-72B-restate/filtered_data_2.jsonl"
tokenizer = AutoTokenizer.from_pretrained("/home/test/test03/models/Qwen3-14B")
with open(dataset_dir,encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        reformated_data = {
            "prompt": [{"role": "user", "content": data["prompt"]}],
            "completion": [{"role": "assistant", "content": data["completion"]}]
        }
        with open(save_dir,"a+",encoding="utf-8") as g:
            json.dump(reformated_data,g,separators=(',',':'),ensure_ascii=False)
            g.write('\n')