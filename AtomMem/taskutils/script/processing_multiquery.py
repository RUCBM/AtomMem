import json
import os
import random
from multiprocessing import Pool
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path
import argparse
import random

# Global variables
QAS = None
DOCS = None
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# SQuAD dataset processing
def read_squad(file):
    with open(file) as f:
        data = json.load(f)
        
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
    with open(file) as f:
        data = json.load(f)
    
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
    return total_qas, total_docs

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
    df = pd.DataFrame(write_jsons)
    df.to_parquet(save_dir)
    return write_jsons

def main(args):
    random.seed(args.seed)

    # Train set
    QAS_train, DOCS_train = read_hotpotqa(args.data_file)
    generate_dataset(
        args.data_size,
        args.output,
        args.doc_num,
        QAS_train,
        DOCS_train,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-query dataset from HotpotQA / 2WikiMultihopQA"
    )

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the training dataset",
    )

    parser.add_argument(
        "--data_size",
        type=int,
        default=20000,
        help="Number of training samples to generate",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output name for training set",
    )
    parser.add_argument(
        "--doc_num",
        type=int,
        default=200,
        help="Maximum number of documents per sample",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    main(args)