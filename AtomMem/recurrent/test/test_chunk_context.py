import torch
import pytest
from dataclasses import dataclass
from typing import Any

# -------------- 模拟最小环境 --------------
@dataclass
class MockConfig:
    chunk_size: int = 32

class MockBatch:
    def __init__(self, context_ids):
        self.batch = {"context_ids": context_ids}

class MockAgent:
    def __init__(self, chunk_size):
        self.config = MockConfig(chunk_size=chunk_size)

    def chunk_context(self, gen_batch):
        ctx_ids = gen_batch.batch['context_ids']
        max_len = ctx_ids.shape[1]
        starts = self.config.chunk_size * self.context_step - 100
        ends   = self.config.chunk_size * (self.context_step + 1) + 100
        max_chunk_len = (ends - starts).max().item()
        pos = torch.arange(max_chunk_len, device=ctx_ids.device).unsqueeze(0) + starts.unsqueeze(1)
        mask = (pos >= 0) & (pos < max_len)
        pos = pos.clamp(min=0, max=max_len-1)
        chunks = torch.where(mask,
                     torch.gather(ctx_ids, 1, pos),
                     torch.tensor(-1, dtype=ctx_ids.dtype, device=ctx_ids.device))
        return chunks

# -------------- 测试用例 --------------
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_chunk_context(device):
    B, T = 3, 256
    ctx = torch.arange(B * T).reshape(B, T).to(device)  # 0..255, 每行递增
    agent = MockAgent(chunk_size=32)

    # 给每个样本不同的 step
    agent.context_step = torch.tensor([0, 2, 8], device=device)

    gen_batch = MockBatch(ctx)
    chunks = agent.chunk_context(gen_batch)

    # 检查形状
    assert chunks.ndim == 2
    assert chunks.shape[0] == B

    # 检查越界置零
    for i, step in enumerate(agent.context_step.tolist()):
        start = 32 * step - 100
        end   = 32 * (step + 1) + 100
        expect_len = max(0, min(end, T)) - max(0, start)
        assert (chunks[i] != -1).sum().item() == expect_len

    # 检查内容：首条样本 step==0 应包含 ctx[0, 0:...] 区间
    ref_start = max(0, 32 * 0 - 100)
    ref_end   = min(T, 32 * 1 + 100)
    ref_ids = ctx[0, ref_start:ref_end]
    actual_ids = chunks[0][chunks[0] != -1]
    torch.testing.assert_close(actual_ids, ref_ids)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])