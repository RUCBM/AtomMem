import torch
import pytest
from dataclasses import dataclass
from typing import List
from langchain.schema import Document
from transformers import AutoTokenizer
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
    def __init__(self, context_ids):
        self.batch = {"context_ids": context_ids}
        self.batch["context_length"] = torch.tensor(
            [ids.numel() for ids in context_ids], dtype=torch.long
        )
        self.non_tensor_batch = {"prompt_ids": [[1, 2, 3]] * context_ids.shape[0]}

class MockDataProto:
    def __init__(self, context_ids):
        self.batch = {"context_ids": context_ids, "responses": torch.randint(0, 100, context_ids.shape)}
        self.non_tensor_batch = {"prompt_ids": [[1, 2, 3]] * context_ids.shape[0]}
        self.meta_info = {}

# ---------- 把 MemoryAgent 的最小实现贴进来 ----------
# 只保留 action / update / chunk_context 即可
from recurrent.impls.memory_longcontext import MemoryAgent, MemoryConfig   # 换成你的真实路径

# ---------- 测试 ----------
@pytest.mark.parametrize("device", ["cpu"])
def test_memory_agent(device):
    t1 = time.perf_counter()
    B, T = 3, 256
    ctx = torch.arange(B * T).reshape(B, T).to(device)
    tokenizer = AutoTokenizer.from_pretrained("/home/test/test03/models/Qwen2.5-7B-Instruct")  # 随便一个轻量 tokenizer

    cfg = MemoryConfig(context_key= "context", chunk_size=32, max_memorization_length=64, max_prompt_length=1024, max_chunks=20, max_final_response_length=1024, max_sliding_window_length=2048)
    agent = MemoryAgent(tokenizer, cfg)
    agent.context_step = torch.tensor([0, 2, 8], device=device)

    # 1. 测试 chunk_context
    gen_batch = MockBatch(ctx)
    chunks = agent.chunk_context(gen_batch)
    print(f"[T1] 初始化耗时: {time.perf_counter() - t1:.3f}s")
    print("=== chunk_context 输出形状 ===")
    print("chunks.shape:", chunks.shape)
    print("chunks[0][:10]:", chunks[0][:10])  # 打印前 10 个 token 看看

    # 2. 测试 update
    # 手工造三条 response 分别触发三种动作
    responses = [
        "<add_memory>hello world</add_memory>",
        "<do_nothing></do_nothing>",
        "<update_query>who am i</update_query>"
    ]
    encodings = [tokenizer.encode(r, add_special_tokens=False) for r in responses]
    mock_response = tokenizer.pad(
        {"input_ids": encodings},
        padding=True,
        max_length=10,
        return_tensors="pt"
    )["input_ids"].to(device)

    # 启动 agent
    agent.start(gen_batch, {})
    print("=== 初始状态 ===")
    print("context_step:", agent.context_step)
    print("query:", agent.query)

    # 构造 DataProto
    gen_out = MockDataProto(ctx)
    gen_out.batch["responses"] = mock_response

    agent.active_mask = torch.ones(B, dtype=torch.bool, device=device)  # 全部激活
    t2 = time.perf_counter()
    agent.action()
    print(f"[T2] action 耗时: {time.perf_counter() - t2:.3f}s")
    t3 = time.perf_counter()
    agent.update(gen_out)
    print(f"[T3] update 耗时: {time.perf_counter() - t3:.3f}s")

    print("=== update 后状态 ===")
    print("context_step:", agent.context_step)
    print("query:", agent.query)
    print("action_rate_metric:", gen_out.meta_info.get("action_rate_metric"))
    print("history:", agent.history)

if __name__ == "__main__":
    test_memory_agent("cpu")