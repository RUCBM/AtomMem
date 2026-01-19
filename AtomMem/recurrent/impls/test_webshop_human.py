import torch
from dataclasses import dataclass
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import sys, os, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from memory_webshop import Webshop, MemoryConfig   # 换成你的真实路径

# ---------- Mock 类 ----------
@dataclass
class MockConfig:
    chunk_size: int = 32
    max_memorization_length: int = 64
    context_key = "context"

class MockBatch:
    def __init__(self, instruction):
        self.batch = {"instruction": instruction}
        self.non_tensor_batch = {"index": ["eval_1"]}

class MockDataProto:
    def __init__(self, instruction):
        self.batch = {"instruction": instruction, "responses": None}
        self.non_tensor_batch = {"index": ["eval_1"]}
        self.meta_info = {}

# ---------- Human-in-the-loop ----------
def run_human_loop(device="cpu"):
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
    agent = Webshop(tokenizer, cfg)

    # 假设只有 1 个样本，instruction 模拟成一串 token
    ctx = torch.arange(50).unsqueeze(0).to(device)  # shape [1, 50]
    gen_batch = MockBatch(ctx)
    agent.start(gen_batch, {})

    print("==== Human-in-the-loop Memory Agent ====")
    print("输入命令，比如 <add_memory>xxx</add_memory><web>search[...]</web>")
    print("输入 exit 结束")

    turn = 0
    while not agent.done():
        turn += 1
        # 让 agent 先执行 action（内部可能生成 prompt）
        t1 = time.perf_counter()
        messages, meta_info_gen = agent.action()
        print(tokenizer.decode(messages[0]))
        print(f"[T-action] {time.perf_counter()-t1:.3f}s")

        # 人工输入响应
        response_text = input(f"\n[Turn {turn}] 输入响应指令: ")
        if response_text.strip().lower() in {"exit", "quit"}:
            print("结束交互")
            break

        # 编码成 token 并构造 DataProto
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        mock_response = torch.tensor(response_ids, dtype=torch.long).unsqueeze(0).to(device)

        gen_out = MockDataProto(ctx)
        gen_out.batch["responses"] = mock_response

        # update
        t2 = time.perf_counter()
        agent.update(gen_out)
        print(f"[T-update] {time.perf_counter()-t2:.3f}s")

        # 打印 agent 内部状态
        print("temp_memory:", agent.temp_memory)
        print("sample_index:", agent.sample_index_list[-1])
        print("final_mask:", agent.final_mask_list[-1])
        print("action_rate_metric:", gen_out.meta_info.get("action_rate_metric"))
        print("-" * 50)
    _, _, reward = agent.end()
    print(reward)


if __name__ == "__main__":
    run_human_loop("cpu")
