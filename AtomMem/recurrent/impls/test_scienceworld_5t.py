import torch
import pytest
from dataclasses import dataclass
from typing import List
from langchain.schema import Document
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import time
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ---------- 最小环境 ----------
@dataclass
class MockConfig:
    chunk_size: int = 32
    max_memorization_length: int = 64
    context_key = "context"

class MockBatch:
    def __init__(self, prompt):
        self.batch = {"prompt": prompt}
        self.non_tensor_batch = {"env_name": ["lifespan-longest-lived", "lifespan-longest-lived", "lifespan-longest-lived"],
                                 "var":[5, 6, 7]}

class MockDataProto:
    def __init__(self, prompt):
        self.batch = {"prompt": prompt, "responses": torch.randint(0, 100, prompt.shape)}
        self.non_tensor_batch = {"env_name": ["lifespan-longest-lived", "lifespan-longest-lived", "lifespan-longest-lived"],
                                 "var":[5, 6, 7]}
        self.meta_info = {}

# ---------- 把 MemoryAgent 的最小实现贴进来 ----------
# 只保留 action / update / chunk_context 即可
from memory_scienceworld import SCIMemAgent, MemoryConfig   # 换成你的真实路径

# ---------- 测试 ----------
@pytest.mark.parametrize("device", ["cpu"])
def test_memory_agent_4round(device):
    import time
    t0 = time.perf_counter()

    max_lens = [63, 95, 95]
    ctx_list = [
        torch.arange(max_len).to(device) for max_len in max_lens
    ]
    # 统一 pad 到最长列（96）
    ctx = pad_sequence(ctx_list, batch_first=True, padding_value=0)  # shape [3, 96]
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/test/test03/models/Qwen2.5-7B-Instruct"
    )

    cfg = MemoryConfig(
        context_key="context",
        chunk_size=32,
        max_memorization_length=64,
        max_prompt_length=1024,
        max_chunks=20,
        max_final_response_length=1024,
        max_sliding_window_length=2048,
    )
    agent = SCIMemAgent(tokenizer, cfg)

    gen_batch = MockBatch(ctx)
    agent.start(gen_batch, {})

    # 2 轮硬编码动作
    actions = [
        [
            "<add_memory>round-1-memory</add_memory><update_query>round-1-query</update_query><action>look around</action>",
            "<update_query>round-2-query</update_query><action>open the doors</action>",
            "<action>go to the badroom</action>"
        ],
        [
            "<modify_memory>Document 1: round-1-memory-modified</modify_memory><action>help</action>",
            "<add_memory>round-2-memory</add_memory><action>objects</action>",
            "<action>inventory</action>"
        ]
    ]

    for turn in range(2):
        # 构造 response token
        responses_tokens = [
            tokenizer.encode(a, add_special_tokens=False)[:100]   # 截断
            for a in actions[turn]
        ]

        # 统一 pad 到最长句子长度（或固定 100）
        mock_response = pad_sequence(
            [torch.tensor(ids) for ids in responses_tokens],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device)
        print(mock_response.shape)
        # action
        t1 = time.perf_counter()
        agent.action()
        print(f"[T-action] {time.perf_counter()-t1:.3f}s")

        # 构造 DataProto
        gen_out = MockDataProto(ctx)
        gen_out.batch["responses"] = mock_response

        # update
        t2 = time.perf_counter()
        agent.update(gen_out)
        print(f"[T-update] {time.perf_counter()-t2:.3f}s")

        # 打印状态
        print("sample_index:", agent.sample_index_list[-1])
        print("final_mask:", agent.final_mask_list[-1])
        print("action_rate_metric:", gen_out.meta_info.get("action_rate_metric"))

    print(f"\n[Total] 五轮总耗时: {time.perf_counter()-t0:.3f}s")

if __name__ == "__main__":
    test_memory_agent_4round("cpu")