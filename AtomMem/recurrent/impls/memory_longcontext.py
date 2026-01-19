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
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import re
import gc, psutil, os
import uuid
import random

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
    random_abandon: bool

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

TEMPLATE = """You are presented with a section of an article that may contain the answer to the problems, and a previous memory. Please read the provided section carefully and select memory actions. 

Four kinds of memory actions are available: <do_nothing>, <update_query>,  <add_memory>, <modify_memory>
<do_nothing>: If there is no useful information in current section, use do_nothing and not use any other actions.

<update_query>: The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query. DO NOT repeatly update query, if you don’t have the desired memory, it means the entry does not exist in the knowledge base. You have update query {query_times} times, exceed 3 will lead to the task failure. 

<add_memory>: creates a new entry in the memory. You do not need to repeatly add the memory shown to you or enter index of the memory. 

<modify memory> updates the existing entry, You need to enter the memory index "Memory i:" to specify which memory to modify.

This is the problems you need to solve:
{prompt}

This is current query to retrieve memory:
{query}

This is current memory:
{memory}

This is the article:
{chunk}

Use paired XML tags as action markers so you can perform multiple actions—such as adding several memories—in a single response.

Example 1:
<update_query>The dance partner of Yulia Zagoruychenko.</update_query> 
Example 2:
<do_nothing></do_nothing>
Example 3:
<add_memory>Yulia Zagoruychenko's dance partner is Riccardo</add_memory>
Example 4:
<modify_memory>Memory 1: Yulia Zagoruychenko's dance partner is not Riccardo. He is Emily's dance partner, I need to find another guy.</modify_memory>
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Based on the memory, use '<final_answer></final_answer>' to answer the problem in \\boxed{{}}. You can use update query to get memory. DO NOT repeatly update similar query, if you don’t have the desired memory, it means the entry does not exist in the knowledge base. You have update query {query_times} times, exceed 3 will lead to the task failure.

problem:
{prompt}

Current query to retrieve memory:
{query}

current memory:
{memory}

Output the answer only as an exact match—do not add any description.
Example 1:
<update_query>The dance partner of Yulia Zagoruychenko.</update_query>
Example 2:
<final_answer>\\boxed{{Jacob}}</final_answer>.
"""


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
        self.vectorstore = FAISS(self.call_remote_embedding, self.faiss_index, docstore= InMemoryDocstore(),
                index_to_docstore_id={})
        self.agent_step = [0]*self.bsz
        self.context_step = torch.zeros(self.bsz, dtype=torch.long)
        self.temp_memory = [[]]*self.bsz
        self.query = ["No query"]*self.bsz
        self.query_times = [0]*self.bsz
        self.history = [""]*self.bsz
    
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
                if self.query[idx] != "No query":
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"trajectory_id": idx},"fetch_k": 5000})
                    memory = retriever.invoke(self.query[idx])
                    self.temp_memory[idx] = memory
                    memory_text = "\n".join(f"Memory {i+1}: {doc.page_content}" for i, doc in enumerate(memory))[:self.config.max_memorization_length] if memory else self.NO_MEMORY_TOKENS
                    del retriever
                else:
                    memory_text = self.NO_MEMORY_TOKENS
                # build message
                prompt = self.tokenizer.decode(gen_batch.non_tensor_batch['prompt_ids'][idx],skip_special_tokens=True)
                message = [{"role": "user", "content": TEMPLATE_FINAL_BOXED.format(prompt = prompt, query = self.query[idx], memory=memory_text, query_times=self.query_times[idx])}]
                message = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
                self.messages.append(message)
            # 如果仍有context，则使用常规prompt
            else:
                # get memory
                if self.query[idx] != "No query":
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"trajectory_id": idx},"fetch_k": 5000})
                    memory = retriever.invoke(self.query[idx])
                    self.temp_memory[idx] = memory
                    memory_text = "\n".join(f"Memory {i+1}: {doc.page_content}" for i, doc in enumerate(memory))[:self.config.max_memorization_length] if memory else self.NO_MEMORY_TOKENS
                    del retriever
                else:
                    memory_text = self.NO_MEMORY_TOKENS
                # build message
                prompt = self.tokenizer.decode(gen_batch.non_tensor_batch['prompt_ids'][idx],skip_special_tokens=True)
                message = [{"role": "user", "content": TEMPLATE.format(
                            prompt = prompt,
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
        gc.collect()
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        action_rate_metric = {"action_rate/do_nothing":0,"action_rate/not_follow": 0, "action_rate/update_query": 0, "action_rate/add_memory": 0, "action_rate/modify_memory": 0, "total": gen_output.batch['responses'].shape[0]}
        if not gen_output.meta_info.get("is_validate", False) and self.config.random_abandon:
            print("random_abandon activated!")
            gen_output = self.random_abandon(gen_output)
        if not torch.all(self.is_final):
            true_indices = torch.where(self.active_mask)[0].tolist()
            texts = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            for idx, real_idx in enumerate(true_indices):
                text = self.tokenizer.decode(texts[idx], skip_special_tokens=True)
                self.history[real_idx] += "\n-------------\n" + text
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
        self.dump_vectorstore("/home/test/test03/huoyupeng/MemAgent/MemAgent-main/dataset.txt")
        del self.vectorstore
        del self.faiss_index
        del self.query
        del self.temp_memory
        del self.history
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index, []
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=5000):
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

                elif tag_name == 'update_query':
                    self._handle_query(idx, content.strip(), action_rate_metric)

                elif tag_name == 'do_nothing':
                    self._handle_do_nothing(idx, action_rate_metric)
                    move_on = True

                elif tag_name == 'final_answer':
                    self.is_final[idx] = True
            if move_on:
                self.context_step[idx] += 1
        except Exception as e:
            print(f"fail to parse the response:{e}")
                
    def _handle_add(self, idx: int, content: str, metric: Dict[str, int]):
        self._write_memory_to_vectorstore(idx, content)
        self.query_times[idx] = 0
        metric['action_rate/add_memory'] += 1
        
    def _handle_modify(self, idx: int, content: str, metric: Dict[str, int]):
        # 期望格式：<modify_memory>Document 3: new content</modify_memory>
        m = re.match(r'Memory\s+(\d+)\s*:\s*(.*)', content, flags=re.S)
        if not m:
            metric['action_rate/not_follow'] += 1
            return
        doc_idx = int(m.group(1)) - 1
        new_content = m.group(2).strip()
        if 0 <= doc_idx < len(self.temp_memory[idx]):
            old_content = self.temp_memory[idx][doc_idx].page_content
            self.modify_document(idx, old_content, new_content)
        metric['action_rate/modify_memory'] += 1
        self.query_times[idx] = 0
        
    def _handle_query(self, idx: int, query: str, metric: Dict[str, int]):
        if self.query_times[idx] > 2:
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

    def modify_document(self, traj_id, old_content, new_content):
        id_map = self.vectorstore.index_to_docstore_id
        docstore = self.vectorstore.docstore
        # 先删后加
        for idx, doc_id in id_map.items():
            doc = docstore.search(doc_id)
            if old_content == doc.page_content:
                self.vectorstore.delete([doc_id])
                break
        self._write_memory_to_vectorstore(traj_id, new_content)

    def random_abandon(self, gen_output, p_response=0.15, p_delete=0.4):
        """
        随机弃用 response 中的动作，保持 (B, L) 严格不变并使用左 pad。

        规则:
        - 每个 response 以 p_response 概率被选中修改。
        - 对选中的 response，每个动作以 p_delete 概率删除，否则保留。
        - 如果删除后 response 中没有动作，则补 <do_nothing></do_nothing>。

        Args:
            gen_output: DataProto-like 对象，包含 batch["responses"] (torch.Tensor of shape (B, L))
            tokenizer: tokenizer（需有 .pad_token_id, .encode(), .decode()）
            p_response: 单个 response 被选中操作的概率（默认 0.15）
            p_delete: 单个动作被删除的概率（默认 0.4）
        """
        pattern = re.compile(r"<(add_memory|modify_memory|update_query)>(.*?)</\1>", re.DOTALL)

        responses_tensor = gen_output.batch["responses"]   # shape = (B, L)
        device = responses_tensor.device
        B, L = responses_tensor.shape
        dtype = responses_tensor.dtype

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None; left-pad logic requires a pad token id.")

        # 1) decode 每一行
        responses = []
        for row in responses_tensor:
            ids = row.tolist()
            ids_nopad = [i for i in ids if i != pad_id]
            text = self.tokenizer.decode(ids_nopad, skip_special_tokens=True)
            responses.append(text)

        # 2) 修改逻辑
        new_responses = []
        for resp in responses:
            if random.random() < p_response:
                print(f"original_response:{resp}")
                matches = list(pattern.finditer(resp))
                if not matches:
                    # 没有动作 -> 保持原样
                    new_resp = resp
                else:
                    # 遍历每个匹配，逐个掷色子
                    parts = []
                    last_end = 0
                    kept_any = False
                    for m in matches:
                        # 保留上一次匹配和当前匹配之间的原始文本（非动作部分）
                        parts.append(resp[last_end:m.start()])
                        if random.random() < p_delete:
                            # 删除该动作：跳过
                            pass
                        else:
                            # 保留动作标签
                            parts.append(m.group(0))
                            kept_any = True
                        last_end = m.end()

                    # 保留最后一个动作后的剩余文本
                    parts.append(resp[last_end:])
                    new_resp = "".join(parts)

                    if not kept_any:
                        # 删除了所有动作：<do_nothing>
                        new_resp = "<do_nothing></do_nothing>"
                print(f"new_response:{new_resp}")
            else:
                new_resp = resp

            new_responses.append(new_resp)

        # 3) encode & 右 pad 保持 (B, L)
        encoded_rows = []
        for r in new_responses:
            ids = self.tokenizer.encode(r, add_special_tokens=False)
            if len(ids) > L:
                ids = ids[:L]  # 右 pad 时截取前 L 个 token（保持左侧内容）
            elif len(ids) < L:
                ids = ids + [pad_id] * (L - len(ids))  # 右 pad
            encoded_rows.append(torch.tensor(ids, dtype=dtype, device=device))

        new_tensor = torch.stack(encoded_rows, dim=0)  # (B, L)
        assert new_tensor.shape == responses_tensor.shape

        gen_output.batch["responses"] = new_tensor
        return gen_output
        


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
