import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
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
from langchain.embeddings.base import Embeddings
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

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length + self.max_sliding_window_length
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
        row_dict["prompt"] = chat[0]["content"]
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt"]
    

SYSTEM_PROMPT = """You are presented with a section of an article and a previous memory. Please learn the provided section carefully and select memory actions.

five kinds of memory actions are available:
<update_query>: The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query. 

<add_memory>: creates a new entry in the memory. You do not need to repeatly add the memory shown to you or enter index of the memory. 

<pin_memory>: creates a new entry that will always be retrieved subsequently. You do not need to repeatly pin the same memory. Modify it throught <modify_memory>

<modify_memory>: updates the existing entry, You need to enter the memory index "Memory i:" to specify which memory to modify. To manage large volumes of memory effectively, you should prioritize using the "modify" function MORE frequently, rather than relying solely on "add" operations.

<delete_memory>: delete a memory. You need to enter the memory index "Memory i" to specify which memory to delete. You must delete duplicate memory entries!

Use paired XML tags as action markers so you can perform multiple actions—such as adding several memories—in a single response.
Important: Give reasoning before take action, including the rule check. Especially when update query, check current query times.

action example 1:
<update_query>dance partner; Yulia Zagoruychenko.</update_query> 
action example 2:
<add_memory>
Document 10 indicates that the dance event took place in Moscow in October and that Yulia participated in it. I need to focus more on who else attended this event or who traveled to Moscow in October, in order to infer who Yulia’s dance partner might be.
</add_memory>
action example 3:
<pin_memory>The current goal is to browse the documents and identify people or events related to Moscow.</pin_memory>
action example 4:
<modify_memory>Memory 1: The current article provides updated competition records showing that Riccardo Cocchi is now partnered with Emily in the 2025 season, while no recent evidence confirms his continued partnership with Yulia Zagoruychenko. This conflicts with the previous memory stating that Yulia’s partner is Riccardo. Since the information clearly supersedes the earlier record, the correct action is to modify the existing memory to reflect that Yulia’s current dance partner is unknown as of 2025, and mark the entry for re-verification.</modify_memory>
action example 5:
It can be observed that Entry 2 and Entry 6 are largely duplicated. Since Entry 6 is more recent, I choose to delete Entry 2. <delete_memory>Memory 2</delete_memory>
action example 6:
do nothing
"""

TEMPLATE_MEMORY = """
This is the question you need to solve:
{prompt}

This is current query to retrieve memory:
{query}

This is current memory related to the query:
{memory}

Tips:
 -DO NOT repeatly update query. If you don’t have the desired memory, it means the entry does not exist in the knowledge base. You have update query **{query_times}** times, exceed **3** will lead to the task failure. AVOID using update_query multiple times within a single response, instead, you can use a long and composite query to retrieve document of different question. The query matches documents based on semantic embeddings, and composite queries are best composed of 2-4 keywords.
 -When you choose update_query, the article will be frozen, meaning you will see the same article again in the next round before selecting your next action. This means that if you want to read the next chunk, stop updating the query.

This is the article:
{chunk}
"""

STRATEGY_INJECTION_1 = """
Important:
Now you should follow this strategy to manage memory:
1) Use pin_memory to permanently add absolutely relevant or highly likely relevant memories in your memory. Consolidate your memory entries so that the number of pin_memory entries does not exceed 3.
2) Use add_memory to load as many other not-so-related memories as possible into your memory, as these can be retrieved later. Because although these memories may currently appear irrelevant, subsequent articles may reveal their relevance. You should segment multiple memories based on semantics, grouping strongly related documents into a single memory. Please compress the descriptions of these memories, Names, attributes, and numbers may be crucial. **select and add at least 5 not-so-related documents in your memory each turn!**
3) Use modify for new information rather than pinning new entries.
Focus on your query times:
You have update query **{query_times}** times, exceed **3** will lead to the task failure.
"""

TEMPLATE_MEMORY = TEMPLATE_MEMORY + STRATEGY_INJECTION_1

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Based on the memory, use '<final_answer></final_answer>' to answer the problem in \\boxed{{}}. You can use update query to get memory.

Tips:
 -You have update query **{query_times}** times, exceed 3 will lead to the task failure. AVOID using update_query multiple times within a single response, instead, you can use a long and composite query to retrieve document of different question. When you choose update_query, the process will be frozen, meaning you can answer the question in the next round.
 -You can use iterative retrieval to pin the important content found through retrieval into memory space, ensuring it remains visible in the next round. Don't give up on searching for answers easily; actively try different queries to explore your memory, and pin the queries you've used to memory to avoid repetition.

problem:
{prompt}

Current query to retrieve memory:
{query}

current memory:
{memory}

Output the answer only as an exact match—do not add any description.
Example 1:
My memory contains no information about Alice's dance partner. I need to first use update_query to search for relevant entries. <update_query>The dance partner of Alice.</update_query>
Example 2:
Based on my memory, I can see that Jacob is Alice's dance partner. I believe we can directly conclude the answer without needing an additional update_query. <final_answer>\\boxed{{Jacob}}</final_answer>.

Focus on your query times:
You have update query **{query_times}** times, exceed **3** will lead to the task failure.
"""

class Qwen3Embedding(Embeddings):
    """LangChain 兼容的 Qwen3 本地 embedding 接口"""
    def __init__(self, base_url: str = "http://localhost:8007/v1/",
                 model_name: str = "qwen3-embedding",
                 api_key: str = "sk-123"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档 embedding"""
        return self._get_embeddings(texts)

    def embed_query(self, query: str) -> List[float]:
        """单条查询 embedding"""
        text = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}"
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(
            input=texts,
            model=self.model
        ).model_dump()
        # 支持多条 / 单条统一返回 List[List[float]]
        if len(resp["data"]) > 1:
            embeddings = [item["embedding"] for item in resp["data"]]
        else:
            embeddings = [resp["data"][0]["embedding"]]
        # 转 list 即可，LangChain 内部不需要 np.array
        return embeddings


class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length
        logger.info(f'\n[RECURRENT] max_input_length: = {self.max_input_length}\n')
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
        embedding = Qwen3Embedding()
        self.vectorstore = FAISS(embedding, self.faiss_index, docstore= InMemoryDocstore(),
                index_to_docstore_id={})
        self.agent_step = [0]*self.bsz
        self.context_step = torch.zeros(self.bsz, dtype=torch.long)
        self.temp_memory = [[] for _ in range(self.bsz)]
        self.pin_memory = [[] for _ in range(self.bsz)]
        self.query = ["defult query, change it!"]*self.bsz
        self.query_times = [0]*self.bsz
        self.prompt = []
        for i in range(self.bsz):
            if not isinstance(gen_batch.non_tensor_batch["prompt"][i], list):
                # maybe ndarray
                self.prompt.append(gen_batch.non_tensor_batch["prompt"][i].tolist())
            else:
                self.prompt.append(gen_batch.non_tensor_batch["prompt"][i])
        self.answer_idx = [0]*self.bsz
        print("self.bsz:",self.bsz)
    
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
                # 此时需要循环多轮，改变prompt，使得模型回答所有问题
                # add pin_memory
                memory_text = "\n".join(f"Memory {i+1}: (Pinned){memory}" for i, memory in enumerate(self.pin_memory[idx])) if self.pin_memory[idx] else ""
                memory_text = memory_text + '\n'
                l = len(self.pin_memory[idx])
                if l<12:
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": 12-l, "filter": {"trajectory_id": idx},"fetch_k": 8000})
                    memory = retriever.invoke(self.query[idx])
                    self.temp_memory[idx] = memory
                    memory_text += "\n".join(f"Memory {i+l+1}: {doc.page_content}" for i, doc in enumerate(memory)) if memory else self.NO_MEMORY_TOKENS
                    del retriever
                memory_text = memory_text[:self.config.max_memorization_length]
                # build message
                message = [{"role": "system", "content": SYSTEM_PROMPT}, 
                           {"role": "user", "content": TEMPLATE_FINAL_BOXED.format(
                            prompt = self.prompt[idx][self.answer_idx[idx]],
                            query = self.query[idx],
                            memory=memory_text,
                            chunk=self.tokenizer.decode(chunk[idx][chunk[idx] != self.tokenizer.pad_token_id],skip_special_tokens=True), # unpadding needed here
                            query_times=self.query_times[idx])}]
                message = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
                self.messages.append(message)
            # 如果仍有context，则使用常规prompt
            else:
                # get memory
                # add pin_memory
                memory_text = "\n".join(f"Memory {i+1}: (Pinned){memory}" for i, memory in enumerate(self.pin_memory[idx])) if self.pin_memory[idx] else ""
                memory_text = memory_text + '\n'
                l = len(self.pin_memory[idx])
                if l<12:
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": 12-l, "filter": {"trajectory_id": idx},"fetch_k": 8000})
                    memory = retriever.invoke(self.query[idx])
                    self.temp_memory[idx] = memory
                    memory_text += "\n".join(f"Memory {i+l+1}: {doc.page_content}" for i, doc in enumerate(memory)) if memory else self.NO_MEMORY_TOKENS
                    del retriever
                memory_text = memory_text[:self.config.max_memorization_length]
                # build message
                message = [{"role": "system", "content": SYSTEM_PROMPT}, 
                           {"role": "user", "content": TEMPLATE_MEMORY.format(
                            prompt = '\n'.join(self.prompt[idx]),
                            query = self.query[idx],
                            memory=memory_text,
                            chunk=self.tokenizer.decode(chunk[idx][chunk[idx] != self.tokenizer.pad_token_id],skip_special_tokens=True), # unpadding needed here
                            query_times=self.query_times[idx])}]
                message = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
                self.messages.append(message)
        self.meta_info = {'input_pad_to': self.max_input_length,
                    'pad_to': self.config.gen_pad_to,
                    'generation_kwargs': {
                    'max_tokens': self.config.gen_max_tokens_memorization,
                    'n': 1 # note that we have already repeat n times in ray_trainer
                    }}
        logger.info(f'MemoryAgent.action() done')
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        action_rate_metric = {"action_rate/do_nothing":0,"action_rate/not_follow": 0, "action_rate/update_query": 0, "action_rate/add_memory": 0, "action_rate/modify_memory": 0, "action_rate/pin_memory": 0, "action_rate/delete_memory": 0, "total": gen_output.batch['responses'].shape[0], "average_DB_size" : 0}
        action_rate_metric["average_DB_size"] = len(self.vectorstore.index_to_docstore_id)/self.bsz
        if not torch.all(self.is_final):
            true_indices = torch.where(self.active_mask)[0].tolist()
            texts = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            for idx, real_idx in enumerate(true_indices):
                text = self.tokenizer.decode(texts[idx], skip_special_tokens=True)
                self.parse_action(real_idx, text, action_rate_metric)
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
        self.dump_vectorstore("/nfsdata/huoyupeng/Memory_Agent/dataset.txt")
        del self.vectorstore
        del self.faiss_index
        del self.query
        del self.temp_memory
        del self.prompt
        del self.answer_idx
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        
        return final_mask, sample_index, []
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=8000):
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
        if torch.any(self.active_mask):
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")
            
    def parse_action(self, idx: int, text: str, action_rate_metric: Dict[str, int]):
        """
        支持在一次响应里出现多个 <tag>…</tag> 动作。
        按照“先匹配先执行”原则处理；如无任何合法标签则记 not_follow。
        """
        # 1. 提取所有成对标签区间
        try:
            tags = self._extract_paired_tags(text)
            if not tags:
                action_rate_metric['action_rate/not_follow'] += 1
                self.context_step[idx] += 1
                return
            move_on = False
            for tag_name, content in tags:
                tag_name = tag_name.lower()
                if tag_name == 'add_memory':
                    self._handle_add(idx, content.strip(), action_rate_metric)
                    move_on = True
                elif tag_name == 'modify_memory':
                    self._handle_modify(idx, content.strip(), action_rate_metric)
                    move_on = True
                elif tag_name == 'pin_memory':
                    self._handle_pin(idx, content.strip(), action_rate_metric)
                elif tag_name == 'update_query':
                    self._handle_query(idx, content.strip(), action_rate_metric)
                elif tag_name == 'delete_memory':
                    self._handle_delete(idx, content.strip(), action_rate_metric)
                elif tag_name == 'do_nothing':
                    self._handle_do_nothing(idx, action_rate_metric)
                    move_on = True
                elif tag_name == 'final_answer':
                    if isinstance(self.prompt[idx], list):
                        if self.answer_idx[idx]< len(self.prompt[idx]) -1:
                            self.answer_idx[idx] += 1
                            self.query_times[idx] = 0
                        else:
                            self.is_final[idx] = True
                    else:
                        self.is_final[idx] = True
            if move_on:
                self.context_step[idx] += 1
        except Exception as e:
            print(f"fail to parse the response:{e}")
                
    def _handle_add(self, idx: int, content: str, metric: Dict[str, int]):
        self._write_memory_to_vectorstore(idx, content)
        self.query_times[idx] = 0
        metric['action_rate/add_memory'] += 1
    
    def _handle_pin(self, idx: int, content: str, metric: Dict[str, int]):
        self.pin_memory[idx].append(content)
        self.query_times[idx] = 0
        metric['action_rate/pin_memory'] += 1
        
    def _handle_modify(self, idx: int, content: str, metric: Dict[str, int]):
        # 期望格式：<modify_memory>Document 3: new content</modify_memory>
        m = re.match(r'Memory\s+(\d+)\s*:\s*(.*)', content, flags=re.S)
        if not m:
            metric['action_rate/not_follow'] += 1
            return
        doc_idx = int(m.group(1)) - 1
        new_content = m.group(2).strip()
        l = len(self.pin_memory[idx])
        if 0 <= doc_idx < l:
            self.pin_memory[idx][doc_idx] = new_content
        if l <= doc_idx < l + len(self.temp_memory[idx]):
            old_content = self.temp_memory[idx][doc_idx-l].page_content
            self.modify_document(idx, old_content, new_content)
        metric['action_rate/modify_memory'] += 1
        self.query_times[idx] = 0

    def _handle_delete(self, idx: int, content: str, metric: Dict[str, int]):
        m = re.match(r'Memory\s+(\d+)', content, flags=re.S)
        if not m:
            metric['action_rate/not_follow'] += 1
            return
        doc_idx = int(m.group(1)) - 1
        l = len(self.pin_memory[idx])
        if 0 <= doc_idx < l:
            self.pin_memory[idx].pop(doc_idx)
        elif l <= doc_idx < l+len(self.temp_memory[idx]):
            old_content = self.temp_memory[idx][doc_idx-l].page_content
            self.modify_document(idx, old_content, "")
        metric['action_rate/delete_memory'] += 1
        self.query_times[idx] = 0
        
    def _handle_query(self, idx: int, query: str, metric: Dict[str, int]):
        if self.query_times[idx] > 2:
            if self.answer_idx[idx]< len(self.prompt[idx]) -1:
                self.answer_idx[idx] += 1
                self.query_times[idx] = 0
            else:
                self.is_final[idx] = True
            return
        if query: 
            self.query[idx] = query
        metric['action_rate/update_query'] += 1
        self.query_times[idx] += 1
        
    def _handle_do_nothing(self, idx: int, metric: Dict[str, int]):
        metric['action_rate/do_nothing'] += 1
        self.query_times[idx] = 0
        
    @staticmethod
    def _extract_paired_tags(text: str):
        """
        返回 [(tag_name, inner_content), ...]，按出现顺序。
        只做最外层匹配，不处理嵌套。
        """
        pattern = re.compile(r'<(\w+)[^>]*>(.*?)</\1>', flags=re.S | re.I)
        return pattern.findall(text)

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

    def modify_document(self, traj_id, old_content, new_content):
        id_map = self.vectorstore.index_to_docstore_id
        docstore = self.vectorstore.docstore
        # 先删后加
        for idx, doc_id in id_map.items():
            doc = docstore.search(doc_id)
            if old_content == doc.page_content:
                succ = self.vectorstore.delete([doc_id])
                if not succ:
                    print("delete fail!")
                break
        self._write_memory_to_vectorstore(traj_id, new_content)
        


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
