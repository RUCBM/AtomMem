# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import random
from multiprocessing import Pool
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path

# Global variables
QAS = None
DOCS = None
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# SQuAD dataset processing
def read_squad(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
        
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
    return total_qas, total_docs

# HotpotQA dataset processing
def read_hotpotqa(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))
    
    total_docs = [f"{p['title']}\n{''.join(p['paragraph_text'])}" for d in data for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{p['title']}\n{''.join(p['paragraph_text'])}"] for p in d['paragraphs']],
        })
    return total_qas, total_docs

# def generate_random_chunks(total_length):
#     """生成随机大小的分组"""
#     chunks = []
#     current = 0
#     while current < total_length:
#         values = [1] + list(range(2, 11))
#         weights = [0.5] + [0.5/9] * 9
#         chunk_size = 16
        
#         end = min(current + chunk_size, total_length)
#         chunks.append((current, end))
#         current = end
    
#     return chunks

def generate_random_chunks(total_length):
    """生成随机大小的分组"""
    chunks = []
    current = 0
    while current < total_length:
        values = [1] + list(range(2, 11))
        weights = [0.5] + [0.5/9] * 9
        chunk_size = random.choices(values, weights=weights, k=1)[0]
        
        end = min(current + chunk_size, total_length)
        chunks.append((current, end))
        current = end
    
    return chunks

def generate_input_output(index, num_docs):
    global QAS, DOCS
    curr_q = [QAS[i]['query'] for i in range(index[0], index[1])]
    curr_a = [QAS[i]['outputs'] for i in range(index[0], index[1])]
    curr_docs = []
    curr_more = []
    for i in range(index[0], index[1]):
        curr_docs.extend(QAS[i]["context"])
        # curr_more.extend(QAS[index].get('more_context', []))

    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS
    
    random.Random(4).shuffle(all_docs)
    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    
    formatted_output = {
        "data_source": "multiquery-hotpotqa",
        "prompt": [{
            "role": "user",
            "content": curr_q,
        }],
        "context": context,
        "ability": "memory",
        "reward_model": {
            "style": "rule",
            "ground_truth": curr_a
        },
        "extra_info": {
            'index': index,
            "question": curr_q,
            "num_docs": num_docs,
        }
    }
    return formatted_output

def generate_dataset(num_samples: int, save_dir: str, incremental: int = 10, qas=None, docs=None):
    global QAS, DOCS
    if qas is None or docs is None:
        raise ValueError("QAS and DOCS must be provided.")
    
    QAS = qas
    DOCS = docs
    
    length = min(num_samples, len(QAS))
    print("start")
    
    from utils import TqdmExecutor
    random_chunk = generate_random_chunks(num_samples)
    write_jsons = TqdmExecutor(max_workers=os.cpu_count()).run(generate_input_output, random_chunk, num_docs=incremental)
    write_jsons = write_jsons[:200]
    # tokens = [len(x) for x in tokenizer([j['context'] for j in write_jsons])['input_ids']]
    # print(max(tokens), min(tokens), sum(tokens) / len(tokens))
    # Save to Parquet file
    df = pd.DataFrame(write_jsons)
    df.to_parquet(save_dir + ".parquet")
    return write_jsons

if __name__ == "__main__":
    random.seed(42)
    
    # QAS_train, DOCS_train = read_hotpotqa('musique_ans_v1.0_train.jsonl')
    # generate_dataset(10000, 'musique_multiquery_train', 200, QAS_train, DOCS_train)
    QAS_dev, DOCS_dev = read_hotpotqa('musique_ans_v1.0_dev.jsonl')
    generate_dataset(1000, 'musique_multiquery_dev_400', 400, QAS_dev, DOCS_dev)

# if __name__ == "__main__":
    # random.seed(42)
    
    # QAS_train, DOCS_train = read_hotpotqa('2wikimultihop_train.json')
    # generate_dataset(10000, '2wikimultihop_train_process', 200, QAS_train, DOCS_train)
    # QAS_dev, DOCS_dev = read_hotpotqa('2wikimultihop_dev.json')
    # generate_dataset(200, '2wikimultihop_dev_process', 200, QAS_dev, DOCS_dev)