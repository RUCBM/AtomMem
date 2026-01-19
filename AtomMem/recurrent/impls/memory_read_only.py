import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, IncrementalChatTemplate, chat_template, now, unpad
from verl.protocol import DataProto
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import re
import gc, psutil, os
import uuid
import warnings

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens3
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length
    max_sliding_window_length: int
    # sliding_window之内的context保留

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids"]

SYSTEM_PROMPT = """You are presented with a problem: {prompt}. take action and solve it."""

TEMPLATE = """You are presented with a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and select an action based on the section: 1) query 2) do nothong 3) update memory. 
The system maintains a query for memory retrieval; the query should contain the information you currently need, your motivation, etc. You can modify it via query operations. You will not get any memory unless you give a query.
<query>
{query}
<query>

do nothing and update memory are used to decide whether to update the memory bank, if you choose to use them, be sure to retain all relevant details that may be useful in the future.

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Always use one of the exact prefixes as the action indicator, such as "query:", "do nothing" and "update memory:".

Example 1:
query: Find the dance partner of Yulia Zagoruychenko. 
Example 2:
do nothing
Example 3:
update memory: Yulia Zagoruychenko's dance partner is Riccardo.
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Based on the memory, use 'final answer:' to answer the problem in \\boxed{{}}.

<memory>
{memory}
</memory>

Example:
final answer: ..., therefore the answer is \\boxed{{}}.
"""


class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # we assume that final_message template is difinately shorter than message_template
        logger.info(f'\n[RECURRENT] max_input_length: = {self.config.max_sliding_window_length}\n')
        logger.info(f"chunk_size: {self.config.chunk_size}")
        self.NO_MEMORY_TOKENS = "No previous memory"
    
    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        self.ctx_length = gen_batch.batch['context_length'] # if all context is used, then the sample will no more be active
        self.bsz = len(self.ctx_length)
        self.final_turn = False
        self.is_final = torch.zeros(self.bsz, dtype=torch.bool)
        self.faiss_index = faiss.IndexFlatL2(1024)
        self.vectorstore = FAISS(self.call_remote_embedding, self.faiss_index, docstore= InMemoryDocstore(),
                index_to_docstore_id={})
        self.agent_step = [0]*self.bsz
        self.context_step = torch.zeros(self.bsz, dtype=torch.long)
        self.temp_memory = [[]]*self.bsz
        self.query = ["No query"]*self.bsz
        self.query_times = [0]*self.bsz
        prompt = gen_batch.non_tensor_batch['prompt_ids']
        self.context_sliding_windows = [
            IncrementalChatTemplate(
                self.tokenizer,
                self.config.max_sliding_window_length,
                system=SYSTEM_PROMPT.format(prompt = self.tokenizer.decode(prompt[i], skip_special_tokens=True))
            ) for i in range(self.bsz)]
    
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # suppose 0 is pad_token_id
        # max_chunks = 3, chunk_size = 2
        # pi is token in prompt, ti is token in chat template, 
        # [1,2] [3,4] [5,0] | p0 string
        # [1,2] [3,0] [0,0] | p1,p1 string
        # [1,0] [0,0] [0,0] | p2,p2,p2 string
        # -------- round 1 ---------
        # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
        # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
        # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
        # get mask & positionids
        self.active_mask = ~self.is_final # final -> not active
        context_mask = self.ctx_length > self.context_step * self.config.chunk_size
        gen_batch = self.gen_batch
        self.messages = []
        active_indices = torch.where(self.active_mask)[0]
        chunk = self.chunk_context(gen_batch)
        for idx in active_indices:
            # 如果当前已无context可用，则使用final prompt
            if not context_mask[idx]:
                # get memory
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"trajectory_id": idx},"fetch_k": 5000})
                if self.query[idx] != "No query":
                    memory = retriever.invoke(self.query[idx])
                    memory_text = "\n".join(doc.page_content for doc in memory)[:self.config.max_memorization_length] if memory else self.NO_MEMORY_TOKENS
                else:
                    memory_text = self.NO_MEMORY_TOKENS
                # build message
                message = TEMPLATE_FINAL_BOXED.format(query = self.query[idx], memory=memory_text)
                self.context_sliding_windows[idx].append(self.tokenizer, "user", message)
                tensor_messages = self.context_sliding_windows[idx].build(self.tokenizer)
                self.messages.append(tensor_messages)
            # 如果仍有context，则使用常规prompt
            else:
                # get memory
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"trajectory_id": idx},"fetch_k": 5000})
                if self.query[idx] != "No query":
                    memory = retriever.invoke(self.query[idx])
                    memory_text = "\n".join(doc.page_content for doc in memory)[:self.config.max_memorization_length] if memory else self.NO_MEMORY_TOKENS
                else:
                    memory_text = self.NO_MEMORY_TOKENS
                # build message
                message = TEMPLATE.format(
                            query = self.query[idx],
                            memory=memory_text,
                            chunk=self.tokenizer.decode(chunk[idx][chunk[idx] != self.tokenizer.pad_token_id],skip_special_tokens=True), # unpadding needed here
                    )
                self.context_sliding_windows[idx].append(self.tokenizer, "user", message)
                tensor_messages = self.context_sliding_windows[idx].build(self.tokenizer)
                self.messages.append(tensor_messages)
        self.meta_info = {'input_pad_to': self.config.max_sliding_window_length,
                    'pad_to': self.config.gen_pad_to,
                    'generation_kwargs': {
                    'max_tokens': self.config.gen_max_tokens_memorization,
                    'n': 1 # note that we have already repeat n times in ray_trainer
                    }}
        logger.info(f'MemoryAgent.action() done')
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        action_rate_metric = {"action_rate/do_nothing":0,"action_rate/not_follow": 0, "action_rate/query": 0, "total": gen_output.batch['responses'].shape[0]}
        if not torch.all(self.is_final):
            true_indices = torch.where(self.active_mask)[0].tolist()
            texts = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            for idx, real_idx in enumerate(true_indices):
                text = self.tokenizer.decode(texts[idx], skip_special_tokens=True)
                self.context_sliding_windows[real_idx].append(self.tokenizer, "assistant", text)
                if "update memory:" in text and "do nothing" not in text and "query:" not in text:
                    start_index = text.find('update memory:')
                    content = text[start_index + len('update memory:'):].strip()
                    self._write_memory_to_vectorstore(real_idx, content)
                    self.context_step[real_idx] += 1
                    self.query_times[real_idx] = 0
                elif "do nothing" in text and "update memory:" not in text and "query:" not in text:
                    action_rate_metric['action_rate/do_nothing'] += 1
                    self.context_step[real_idx] += 1
                    self.query_times[real_idx] = 0
                elif "query:" in text and "update memory:" not in text and "do nothing" not in text:
                    # 如果重复query多次，直接视作失败
                    if self.query_times[real_idx] > 1:
                        self.is_final[real_idx] = True
                    query = text.removeprefix("query:").strip()
                    self.query[real_idx] = query
                    action_rate_metric['action_rate/query'] += 1
                elif "final answer:" in text:
                    self.is_final[real_idx] = True
                else:
                    self.temp_memory[real_idx] = []
                    action_rate_metric['action_rate/not_follow'] += 1
                    self.context_step[real_idx] += 1
                self.agent_step[real_idx] += 1
                # 超出最大步数直接结束，防止无限循环
                if self.agent_step[real_idx] > self.config.max_chunks:
                    self.is_final[real_idx] = True
        sample_index = torch.arange(self.bsz, dtype=torch.long)[self.active_mask] # map active sample to original batch
        self.sample_index_list.append(sample_index)
        final_mask = self.is_final[self.active_mask]
        self.final_mask_list.append(final_mask)
        gen_output.meta_info["action_rate_metric"] = action_rate_metric
        self.log_step(gen_output)
        return gen_output
    
    @override
    def done(self):
        return torch.all(self.is_final)
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.messages
        self.dump_vectorstore("/home/test/test03/huoyupeng/MemAgent/MemAgent-main/dataset.txt")
        del self.vectorstore
        del self.faiss_index
        del self.query
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=2000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.agent_step[:10] if not torch.all(self.is_final) else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] Agent STEP {step}{'='*30}")
        context_step = self.context_step[:10] if not torch.all(self.is_final) else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] Context STEP {context_step}{'='*30}")

        # Message and Response section
        if self.active_mask[0]:
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")

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
                     torch.tensor(self.tokenizer.pad_token_id, dtype=ctx_ids.dtype, device=ctx_ids.device))
        # 注意将id=pad_token_id的位置后续mask掉
        return chunks

    def _write_memory_to_vectorstore(self, idx: int, content: str):
        """把 temp_memory[idx] 内容写入向量库并清空该位置"""
        if not content.strip():
            return
        doc = Document(
            page_content=content.strip(),
            metadata={
                "doc_id": str(uuid.uuid4()),
                "trajectory_id": idx
                }
        )
        self.vectorstore.add_documents([doc])
        self.temp_memory[idx] = []        # 置空

    def call_remote_embedding(self, texts):
        base_url = 'http://localhost:8007/v1/'
        model_name = 'qwen3-embedding'
        client = OpenAI(api_key='sk-123', base_url=base_url)
        response = client.embeddings.create(
            input=texts,
            model=model_name,
        ).model_dump()
        if len(response['data']) > 1:
            embeddings = [item['embedding'] for item in response['data']]
        else:
            embeddings = response['data'][0]['embedding']
        np_embeddings = np.array(embeddings, dtype=np.float32)
        return np_embeddings
    
    def dump_vectorstore(self, output_path=None):
        id_map = self.vectorstore.index_to_docstore_id
        docstore = self.vectorstore.docstore

        lines = []
        for idx, doc_id in id_map.items():
            doc = docstore.search(doc_id)
            content = doc.page_content
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            line = f"Index: {idx}\nDoc ID: {doc_id}\nContent: {content}\nMetadata: {metadata}\n{'-'*40}"
            lines.append(line)

        result = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"[INFO] Vectorstore dump written to: {output_path}")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
