import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Type

import sys
sys.path.append('/home/test/test03/huoyupeng/MemAgent/MemAgent-main')

import torch
from codetiming import Timer
import asyncio
import aiohttp
from openai import AsyncOpenAI
from transformers import PreTrainedTokenizer

from verl import DataProto
import hydra
import json

from recurrent.interface import RAgent, RConfig
from recurrent.utils import (chat_template, create_attention_mask, create_position_ids,
                    graceful_padding, indexing_proto,
                    pad_tensor_list_to_length)
from verl.utils import hf_processor, hf_tokenizer
from recurrent.utils import final_batch
from verl.trainer.ppo.reward import load_reward_manager, compute_reward
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.rl_dataset import collate_fn
import ray
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai
import numpy
import re

logger = logging.getLogger(__file__)
logger.setLevel('WARNING')



@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timing_raw.get(name, 0.) + timer.last
    
class InferenceLLM:
    def __init__(self, model_name, base_url):
        self.model_name = model_name
        self.api_key = "sk-123"
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    @retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),  # 1, 2, 4, 8 ...
    stop=stop_after_attempt(5),                         # 最多 5 次
    retry=retry_if_exception_type(
        (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
    ),
    reraise=True,  # 重试耗尽后把最后一次异常抛出来
    )
    async def generate_sequences(self, messages: List[str]) -> Dict:
        """
        与 self.actor_rollout_wg.generate_sequences(batch) 对齐的异步接口。
        输入/输出的 DataProto 结构完全一致。
        """

        # 2. 并行调用 API
        coros = [self._chat(prompt) for prompt in messages]
        answers: List[str] = await asyncio.gather(*coros)
        output_batch = {"prompts": messages, "responses": answers}
        # 3. 构造输出 DataProto
        return output_batch

    async def _chat(self, prompt: str) -> str:
        """单条 prompt -> 文本回答"""
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.6,
        )
        response = resp.choices[0].message.content
        return response if response else ""
        
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: RConfig,
        agent_cls: Type[RAgent],
        model_name: str,
        base_url: str
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.agent = agent_cls(tokenizer, config)
        self.inferencelm = InferenceLLM(model_name=model_name, base_url=base_url)

    
    def generate_sequence(self, messages: List[torch.Tensor]):

        """
        batch may not be divisible by wordsize.
        Use "Hello" as padding, insert padding data into batch so that data 
        """
        message_list = []
        for idx in range(len(messages)):
            message_list.append(self.tokenizer.decode(messages[idx], skip_special_tokens=True))
        output_batch = asyncio.run(self.inferencelm.generate_sequences(message_list))
        return output_batch

    def run_llm_loop(self, gen_batch, timing_raw) -> Tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        """Run main LLM generation loop.
        genbatch: 'context_ids','context_length','prompt_ids'
        timing_raw: timing dict used in ray_trainer, note that we will accumulate the time cost in this loop, instead of override each time as in ray_trainer.
        see `_timer` implementation at the top of this file for more details.
        """
        active_num_list = [] # trace the active number of sample in each turn
        gen_output_list = [] # store I/O batch in each turn, used for policy optimization
        meta_info = gen_batch.meta_info #  do_sample, is_validate, eos/pad are stored in here.
        self.agent.start(gen_batch, timing_raw)
        # Main generation loop, agent should indicate whether to stop
        while not self.agent.done():
            with _timer('mt_prepare', timing_raw):
                messages, meta_info_gen = self.agent.action()
                meta_info_gen.update(meta_info)
                active_num_list.append(len(messages))
                logger.info(f'padding done')
            with _timer('mt_gen', timing_raw):
                output_batch = self.generate_sequence(messages)
                # construct output_batch to tensor, input pad to 8192, output pad to 1024
                prompt_enc = self.tokenizer(
                    output_batch["prompts"],
                    padding="max_length",
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt",
                )
                
                answer_enc = self.tokenizer(
                    output_batch["responses"],
                    padding="max_length",
                    truncation=True,
                    max_length=4096,
                    return_tensors="pt",
                )
                attention_mask = torch.cat([prompt_enc["attention_mask"], answer_enc["attention_mask"]], dim=1)
                gen_output = DataProto.from_dict(tensors={"responses": answer_enc["input_ids"], "prompts": prompt_enc["input_ids"], "attention_mask": attention_mask})
                print(gen_output)
                logger.info('generation done')
            with _timer('mt_update', timing_raw):
                gen_output = self.agent.update(gen_output)
                gen_output_list.append(gen_output)
                logger.info('agent update done')
        final_mask, sample_index, reward = self.agent.end()
        # convert the total number to rate
        # OK, now we've got all we need in gen_output_list, and the final_mask indicates which one is final answer.
        assert len(sample_index) == sum(active_num_list)
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        concat_gen_output = DataProto.concat(gen_output_list)
        if reward:
            concat_gen_output.meta_info["reward"] = reward
        return concat_gen_output, final_mask, sample_index # pyright: ignore
    
# 编写一个函数，计算gen_batch_output中trajectory的reward，并将其filter出来
def compute_score(gen_batch_output, tokenizer: PreTrainedTokenizer, reward_fn):
    reward_batch = gen_batch_output
    reward_tensor, score_tensor, reward_extra_infos_dict = compute_reward(reward_batch, reward_fn)
    # extent reward_tensor to step-level
    score = reward_tensor.sum()
    total = reward_tensor.shape[0]
    return score, total


 
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)    
def eval(config):
    from recurrent.interface import RRegister
    enabled_conf = getattr(config.recurrent, config.recurrent.enable)
    recurrent_register = RRegister.from_filename(enabled_conf.path, enabled_conf.name)
    conf = dict(enabled_conf.config) if enabled_conf.config is not None else {}
    recurrent_config = recurrent_register.config_cls(**conf)
    log_path = config.log_path
    model_name = config.model_name
    base_url = config.base_url
    # 加载参数
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
    processor = hf_processor(config.actor_rollout_ref.model.path, use_fast=True)
    
    generation_manager = LLMGenerationManager(tokenizer=tokenizer,
                    config=recurrent_config,
                    agent_cls=recurrent_register.agent_cls,
                    model_name=model_name,
                    base_url=base_url)
    # 加载数据集
    train_dataset = recurrent_register.dataset_cls(
                recurrent_config=recurrent_config,
                data_config=config.data,
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                processor=processor,
            )
    
    sampler = SequentialSampler(data_source=train_dataset)
    train_dataloader = StatefulDataLoader(dataset=train_dataset,
                                            batch_size=config.data.train_batch_size,
                                            num_workers=8,
                                            drop_last=False,
                                            collate_fn=collate_fn,
                                            sampler=sampler)
    # 加载reward函数
    ray.init()
    EM_reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    LLM_reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {"use_llm":True, "api_key": "sk-123", "base_url": "https://api.deepseek.com", "model_name": "deepseek-chat"}))
    total_LLM = 0
    score_LLM = 0
    total_EM = 0
    score_EM = 0
    for batch_dict in train_dataloader:
        timing_raw = {}
        batch = DataProto.from_single_dict(batch_dict)
        batch_keys_to_pop, non_tensor_batch_keys_to_pop = train_dataset.get_bactch_keys()
        gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
        gen_batch_output, final_mask, sample_index = generation_manager.run_llm_loop(gen_batch, timing_raw)
        gen_batch_output.batch['final_mask'] = final_mask
        gen_batch_output.batch['sample_index'] = sample_index
        gen_batch_output = gen_batch_output.union(batch[sample_index])
        sub_score_LLM, sub_total_LLM = compute_score(gen_batch_output, tokenizer, LLM_reward_fn)
        score_LLM += sub_score_LLM
        total_LLM += sub_total_LLM
        sub_score_EM, sub_total_EM = compute_score(gen_batch_output, tokenizer, EM_reward_fn)
        score_EM += sub_score_EM
        total_EM += sub_total_EM
    with open(log_path, 'w', encoding="utf-8") as f:
        f.write("=======================\n")
        f.write(f"Exact Match Score: {score_EM}\n")
        f.write(f"Exact Match Total:{total_EM}\n")
        f.write("=======================\n")
        f.write(f"LLM Judge Score: {score_LLM}\n")
        f.write(f"LLM Judge Total:{total_LLM}\n")
        f.write(f"Judge Model: DeepSeek-Chat\n")

if __name__ == "__main__":
    eval()