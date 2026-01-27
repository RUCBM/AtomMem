# AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation
> This repo is the official code implementation of **AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation** <br>
<!-- > [[arXiv]](https://arxiv.org/abs/2601.08323) <br> -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

⭐️ Please star this repository if you find it helpful!

## 👀Overview
![main architecture of AtomMem](./assets/main_figure.png)
We introduce **AtomMem**, a dynamic memory framework that reframes memory management as a learnable decision-making problem. Instead of relying on predefined pipelines, AtomMem decomposes memory manipulation into **atomic CRUD operations**—Create, Read, Update, and Delete—and trains the agent to decide when and how to invoke these operations based on task context.

## 🧩Quick Start
### Environment Preparation 

#### clone repository
```
git clone (The URL has been anonymized.)
cd AtomMem
```
#### create a new environment (optional)
```
conda create -n AtomMem python=3.11
conda activate AtomMem
```

#### install dependencies
**Note:** First install torch based on your requirement, torch 2.7.1 is verified, then
```
pip install -r requirements.txt
```

### Data Preparation

**1.** download the dataset in our paper(or prepare your own dataset):

[[HotpotQA]](https://huggingface.co/datasets/hotpotqa/hotpot_qa) [[2WikiMultiHopQA]](https://github.com/Alab-NII/2wikimultihop) [[Musique]](https://github.com/stonybrooknlp/musique)

Place the dataset under the **taskutils/memory_data** directory, organized in the following structure:
```
- taskutils
  |- memory_data
    |- hotpotqa_train.json
    |- hotpotqa_dev.json
```
**2.** run the following command of specific dataset.
```
./taskutils/script/HotpotQA.sh
./taskutils/script/2Wiki.sh
./taskutils/script/Musique.sh
```

### Inference

**1.** download our model from huggingface or modelscope
```
(The URL has been anonymized.)
```

**2.** start the AtomMem vllm service on port 8001
```
vllm serve your_model_path --served_model_name AtomMem --port 8001
```

**3.** start the embedding vllm service on port 9007
```
CUDA_VISIBLE_DEVICES=1 vllm serve your_embedding_model_path --served_model_name qwen3-embedding --task_embed --port 9007
```

**4.** run the following script to inference on our tasks
```
./script/eval/run_eval.sh
```

### Supervised Fine-Tuning

**1.** run the inference script to collect trajectory from stronger model
```
./SFT/inference_script/run_sft_inference.sh
```

**2.** Run the training script to perform SFT using the TRL library
```
./SFT/train_script/run.sh
```

**3.** convert the format of checkpoint from .pt to .safetensor
```
./SFT/train_script/convert.sh
```

### Reinforcement Learning
We build upon Verl v0.2.0 and extend the framework to support multi-turn reinforcement learning. The core components are organized as follows:
```
|- recurrent   (implementation of the agent)
|- verl        (RL algorithms and training pipeline)
```

To quickly run RL training, please execute the following script:

**1.** Start the embedding vllm service on port 9007
```
vllm serve your_embedding_model_path --served_model_name qwen3-embedding --task_embed --port 9007
```
**Note:** We recommend creating a new environment and installing vLLM to launch embedding service, ensuring that the vLLM version is **below 0.8.5**. Otherwise, on an 8-GPU server, the embedding model may fully occupy one GPU, leaving only 7 GPUs available for training.

**2.** run training script
```
./script/train/run_memory_longcontext.sh
```

## 🙏 Acknowledgements
This code is adapted from the [verl](https://github.com/verl-project/verl) framework and the [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent) implementation. We sincerely appreciate their contributions.