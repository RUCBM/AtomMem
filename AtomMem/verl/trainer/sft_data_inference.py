import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Type

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
    def __init__(self, api_key, base_url, model_name):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    async def generate_sequences(self, messages: List[str]) -> Dict:
        """
        An asynchronous interface aligned with self.actor_rollout_wg.generate_sequences(batch).
        The input and output DataProto structures are exactly the same.
        """

        # Call API
        coros = [self._chat(prompt) for prompt in messages]
        answers: List[str] = await asyncio.gather(*coros)
        output_batch = {"prompts": messages, "responses": answers}
        return output_batch

    @retry(
    wait=wait_exponential(multiplier=5, min=5, max=5),  # 1, 2, 4, 8 ...
    stop=stop_after_attempt(10),                         # 最多 10 次
    retry=retry_if_exception_type(
        (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.PermissionDeniedError
        )
    ),
    reraise=True,  # Raise the last exception after all retry attempts are exhausted.
    )
    async def _chat(self, prompt: str) -> str:
        """单条 prompt -> 文本回答"""
        resp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.6,
        )
        message = resp.choices[0].message
        # Qwen format
        raw_output = []

        # Add reasoning content (if any)
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            raw_output.append(f"<think>\n{message.reasoning_content}\n</think>")

        # add content
        if message.content:
            raw_output.append(message.content)

        # add tool_calling (if any)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function = tool_call.function
                raw_output.append(
                    f'<tool_call>\n{{"name": "{function.name}", "arguments": {function.arguments}}}\n</tool_call>'
                )

        final_raw_string = "\n".join(raw_output)
        return final_raw_string if final_raw_string else ""
        
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: RConfig,
        agent_cls: Type[RAgent],
        inference_api_key,
        inference_base_url,
        inference_model_name
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.agent = agent_cls(tokenizer, config)
        self.inferencelm = InferenceLLM(inference_api_key, inference_base_url, inference_model_name)

    
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
                    max_length=16384,
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
        # OK, now we've got all we need in gen_output_list, and the final_mask indicates which one is final answer.
        assert len(sample_index) == sum(active_num_list)
        logger.info(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        concat_gen_output = DataProto.concat(gen_output_list)
        if reward:
            concat_gen_output.meta_info["reward"] = reward
        return concat_gen_output, final_mask, sample_index # pyright: ignore
    
# 编写一个函数，计算gen_batch_output中trajectory的reward，并将其filter出来
def filter_data(gen_batch_output, tokenizer: PreTrainedTokenizer, reward_fn, acc_record):
    reward_batch = gen_batch_output
    reward_tensor, score_tensor, reward_extra_infos_dict = compute_reward(reward_batch, reward_fn)
    acc_record["correct"] += int(torch.sum(reward_tensor))
    acc_record["total"] += int(reward_tensor.shape[0])
    # extent reward_tensor to step-level
    reward_tensor = reward_tensor[gen_batch_output.batch['sample_index']]
    data = []
    print(reward_tensor.shape)
    for idx in range(reward_tensor.shape[0]):
        if torch.sum(reward_tensor[idx]) >= 0.5:
            prompt = tokenizer.decode(gen_batch_output.batch["prompts"][idx],skip_special_tokens=True)
            completion = tokenizer.decode(gen_batch_output.batch["responses"][idx],skip_special_tokens=True)
            data.append({"prompt": [{"role": "user", "content": prompt}],"completion": [{"role": "assistant", "content": completion}]})
    return data


 
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)    
def generate_sft_data(config):
    from recurrent.interface import RRegister
    enabled_conf = getattr(config.recurrent, config.recurrent.enable)
    recurrent_register = RRegister.from_filename(enabled_conf.path, enabled_conf.name)
    conf = dict(enabled_conf.config) if enabled_conf.config is not None else {}
    recurrent_config = recurrent_register.config_cls(**conf)
    path_to_write = config.file_path
    log_to_write = config.log_path
    # 加载参数
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
    processor = hf_processor(config.actor_rollout_ref.model.path, use_fast=True)
    
    generation_manager = LLMGenerationManager(tokenizer=tokenizer,
                    config=recurrent_config,
                    agent_cls=recurrent_register.agent_cls,
                    inference_api_key=config.inference_api_key,
                    inference_base_url=config.inference_base_url,
                    inference_model_name=config.inference_model_name,
                    )
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
                                            drop_last=True,
                                            collate_fn=collate_fn,
                                            sampler=sampler)
    # 加载reward函数
    ray.init()
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    acc_record = {"correct": 0, "total": 0}
    start_count = 0
    for batch_dict in train_dataloader:
        start_count += len(batch_dict['data_source'])
        if start_count < config.start_index:
            continue
        timing_raw = {}
        batch = DataProto.from_single_dict(batch_dict)
        batch_keys_to_pop, non_tensor_batch_keys_to_pop = train_dataset.get_bactch_keys()
        gen_batch = batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
        gen_batch_output, final_mask, sample_index = generation_manager.run_llm_loop(gen_batch, timing_raw)
        gen_batch_output.batch['final_mask'] = final_mask
        gen_batch_output.batch['sample_index'] = sample_index
        gen_batch_output = gen_batch_output.union(batch[sample_index])
        data = filter_data(gen_batch_output, tokenizer, reward_fn, acc_record)
        with open(path_to_write, "a+", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':'))+'\n')
        with open(log_to_write, "w", encoding="utf-8") as f:
            f.write(json.dumps(acc_record, ensure_ascii=False, separators=(',', ':'))+'\n')
        
    
if __name__ == "__main__":
    generate_sft_data()