#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict
import uuid

import os
import sys
import json
import asyncio
import logging
import argparse
import random
import time
import copy
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import concurrent.futures
from threading import Lock
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import re
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

# 导入MCPManager和扩展的OpenAI客户端
from recurrent.envs.mcp_manager import MCPManager
import verl.utils.torch_functional as verl_F
from torch.nn import functional as F
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
import random

logger = logging.getLogger(__file__)
logger.setLevel('INFO')
date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fh = logging.FileHandler(f'/nfsdata/huoyupeng/Memory_Agent/log/{date}.log', mode='a', encoding='utf-8')
fh.setLevel(logging.INFO)

# 2. 定义格式并绑定
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)

# 3. 关联到 logger
logger.addHandler(fh)

@dataclass
class GAIAMemoryConfig(RConfig):
    context_key: str
    # 一轮的输入大于这个值时会启动Stream Input
    chunk_size: int
    max_chunks: int
    # 叙述性prompt的长度
    max_prompt_length: int  #
    # 原子操作的memory字段的长度
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    # 输出长度
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length
    # recent action的最大长度
    max_sliding_window_length: int
    # --- 通用参数保持不变 ---
    manager_url: str = "http://localhost:8088/mcpapi"
    browser_agent_name: str = "qwen3-4b"
    browser_agent_url: list = ["http://localhost:8001/v1/", "http://localhost:8002/v1/", "http://localhost:8003/v1/", "http://localhost:8004/v1/"]
    max_interactions: int = 10

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_final_response_length

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
        recurrent_config: GAIAMemoryConfig,
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
        instruction = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False, return_tensors="pt"
        )[0]
        
        row_dict["question"] = F.pad(instruction, (0, self.max_prompt_length- instruction.size(0)), value=self.tokenizer.pad_token_id)
        return row_dict
    
    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["question"], []

SYSTEM_PROMPT_QWEN = '''
# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and **use tools** to working through them. At the same time, **manage your own memory** to handle this multi-turn task.

## Memory Actions

Your short-term memory is a step-wise updated summary, while your long-term memory is a vector database that can be updated through atomic operations. 

short-term memory is updated using <update_memory>,you should use this every response.

four kinds of memory actions for database are available:
<update_query>: The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query. 

<add_memory>: creates a new entry in the memory. You do not need to repeatly add the memory shown to you or enter index of the memory. 

<modify_memory>: updates the existing entry, You need to enter the memory index "Memory i:" to specify which memory to modify. To manage large volumes of memory effectively, you should prioritize using the "modify" function MORE frequently, rather than relying solely on "add" operations.

<delete_memory>: delete a memory. You need to enter the memory index "Memory i" to specify which memory to delete. You must delete duplicate memory entries!

Use paired XML tags as action markers so you can perform multiple actions—such as adding several memories—in a single response. Output these actions outside your <think> tags.

<update_query>: The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query.
Example: <update_query>Moscow; The dance partner of Yulia Zagoruychenko.</update_query>

action example 1:
<update_query>dance partner; Yulia Zagoruychenko.</update_query> 
action example 2:
<add_memory>
I searched the url: "http://travel.com". The web page indicates that the dance event took place in Moscow in October and that Yulia participated in it. I need to focus more on who else attended this event or who traveled to Moscow in October, in order to infer who Yulia’s dance partner might be.
</add_memory>
action example 3:
<modify_memory>Memory 1: The current article provides updated competition records showing that Riccardo Cocchi is now partnered with Emily in the 2025 season, while no recent evidence confirms his continued partnership with Yulia Zagoruychenko. This conflicts with the previous memory stating that Yulia’s partner is Riccardo. Since the information clearly supersedes the earlier record, the correct action is to modify the existing memory to reflect that Yulia’s current dance partner is unknown as of 2025, and mark the entry for re-verification.</modify_memory>
action example 4:
It can be observed that Entry 2 and Entry 6 are largely duplicated. Since Entry 6 is more recent, I choose to delete Entry 2. <delete_memory>Memory 2</delete_memory>

Response example:
<think>...</think>
<update_memory>
**Task Decomposition and Memory Update Example:**

**Identify the target municipality**
- Search for a municipality in the Yonne department, within the Arrondissement of Sens, featuring: a garden designed by André Le Nôtre, a medieval church built at the source of the Oreuse river, and documentation in an 1886 publication
- **Searched Web url**: https://en.wikipedia.org/wiki/..
- **Result**: Identified as Villeneuve-sur-Yonne (Source: https://en.wikipedia.org/wiki/..)
- **Status**: Completed

**Find Villeneuve-sur-Yonne's feature**
- **Searched Web url**: https://en.wikipedia.org/wiki/.., https://en.wikipedia.org/wiki/.., https://en.wikipedia.org/wiki/..
- **Result**: All match (Le Nôtre garden, Church of the Assumption built on a spring, military fortifications, 1886 publication)(Source: https://en.wikipedia.org/wiki/..)
- **Status**: Completed

**Identify the heritage preservation society**
- Search for the heritage society responsible for preserving historical monuments at this location
- **Status**: In progress
- **New Info**: The search result show that ...
</update_memory>

<update_query>
Société d'Histoire et d'Archéologie de l'Arrondissement de Sens; Archaeological Society; 1886
</update_query>

<add_memory>
Searched Web url: https://en.wikipedia.org/wiki/...
Result: The Historical and Archaeological Society of the Arrondissement of Sens (founded 1886) is the key heritage organization responsible for preserving and documenting historical monuments in Villeneuve-sur-Yonne, which features: 1) a Remarkable Garden of France designed by André Le Nôtre, 2) a medieval church built on the source of the Oreuse river with military fortifications, and 3) prehistoric and Roman historical sites documented in the society's 1886 publication.
</add_memory>

## Task Strategy

1. **Analyze the user's request** to clarify the task objective, break it down into clear sub-goals, and arrange them in logical order.
2. **Develop a concise step-by-step plan** (e.g., 1., 2., 3.), with each step corresponding to a specific sub-goal, obey tool-use guidelines to solve the task.

## Tool-Use Guidelines
4. **Call at most one tool per step**, prioritizing the tool that best advances the current sub-goal.
5. **Tool Prioritization Rule: To access any online resource via a URL (like http:// or https://), including webpages and online PDFs, you must use the fetch_url tool. The read_file tool should only be used for local file URIs (e.g., /app/data/gaia_validation/...).
7. **After each tool call, stop responding immediately** and wait for user feedback or tool results. Do not assume results or continue analysis.
8. **Adjust your plan promptly when new information or challenges arise**, ensuring all sub-goals are covered and nothing is missed.
9. **Output the final answer in the specified format**.
10. **Tool Use Format**
- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- At most one tool call is allowed per responses.

## Answer Format
- **Answers should be direct and concise**, preferably using single words, numbers with commas and unit, or brief phrases.
Example:<final_answer>\\boxed{{Alice}}</final_answer>
- **Never rely solely on your own knowledge to answer questions without using tools. Your responses should be supported by evidence from the environment.
- **Strictly follow the format requirements**, wrapping the final answer in `<final_answer>\\boxed{}</final_answer>` tags.
'''

MEMORY_MESSAGE_TEMPLATE = """
This is your short term memory from the previous turn.
{short_memory}

This is current query to retrieve memory from database:
{query}

This is current memory related to the query:
{long_memory}
"""

STRATEGY_INJECTION_1 = """
Important:
Now you should follow this strategy to manage memory:
1) Break the task into several sub-tasks and keep them in short-term memory. Whenever a sub-task is completed, add its answer to memory and mark that task as done. Do not always verify the information you have known.
2) You should store only the most important information in short-term memory, while placing more detailed and broadly relevant information into long-term memory. This allows you to supplement missing information by retrieving it from long-term memory when needed.
3) The entries you write into the database using add_memory should contain information that your short-term memory tends to overlook—those with weaker relevance or even information that is entirely unrelated at the moment. The text stored in the database should not contain any reasoning or descriptions of the current state; it should consist solely of factual or knowledge-based information.
4) Use a key-value matching approach to articulate your long-term memory. Record the web pages and queries you've searched in your short-term memory and long-term memory to avoid duplicate visits. Example: 
<add_memory>
Searched Web url: https://en.wikipedia.org/wiki/...
or Used Query: Ancient Egyptian hieroglyph
Result: ...
</add_memory>
5) At least add one memory every step. Delete useless memory proactively or modify the content of memory to newest situation. Proactively update the query in each step to keep your long-term memory strongly relevant to the task.
6) AVOID using update_query multiple times within a single response, instead, you can use a long and composite query to retrieve document of different question. The query matches documents based on semantic embeddings, and composite queries are best composed of keywords.

7) You need to determine your overall strategy according to different task. 
For math problems, perform only single-step reasoning; for information retrieval, carry out multi-step tool calls; and for some tasks, you need to reason during the tool-calling process—such as determining the difference between two variables or finding the maximum value of a variable, etc.
8) FOLLOW the tool call format: 
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Focus on your query times:
You have update query **{query_times}** times, exceed **3** will lead to the task failure.
"""

TEMPLATE_MEMORY = MEMORY_MESSAGE_TEMPLATE + STRATEGY_INJECTION_1

BROWSER_AGENT_PROMPT =  """
Please process the following webpage or local file content and user goal to extract relevant information:

## **Webpage/Local file Content** {raw_content}

## **User Goal**

{context_awareness_prompt}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
4. **Output Length Limit**: Please keep the output within {max_completion_tokens} tokens.

**Final Output Format: You MUST use Markdown with the following headings:**
## Rational
(Your analysis of relevance here)

## Evidence
(Your extracted evidence here)

## Summary
(Your final summary here)
"""

# chunk context替换为browser agent

# CHUNK_CONTEXT_TEMPLATE = """
# Tool Response Chunking Notice
# Your original tool call (do not repeat it): {tool_call_text}
# The tool response is too long and will be delivered in multiple chunks.

# chunk {current_idx} of {total}
# {chunk}

# If you want to continue reading: **Do Not invoking ANY NEW tools**. Especially DO NOT use fetch_url to access the page you have accessed, this will reset your reading progress. You ONLY need to manage your memory through memory actions.
# If you believe there is no potentially useful information in the remaining chunks, invoking a new search will discard all the current remaining chunks.
# """

class Qwen3Embedding(Embeddings):
    """LangChain 兼容的 Qwen3 本地 embedding 接口"""
    def __init__(self, base_url: str = "http://localhost:9007/v1/",
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

class GAIAMemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: GAIAMemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length
        logger.info(f'\n[RECURRENT] max_input_length: = {self.max_input_length}\n')
        self.NO_MEMORY_TOKENS = "No retrieved memory"
        # 从服务器获取工具
        self.mcp_manager = MCPManager(manager_url=self.config.manager_url)
        self.all_tools = []
        self.tools_by_server = {}
        _ = asyncio.run(self.initialize())

    
    @override
    def start(
        self, gen_batch: DataProto, timing_raw: dict):
        # system varible
        self.gen_batch = gen_batch
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        self.bsz = len(gen_batch.batch["question"])
        self.prefix_prompt = [[] for i in range(self.bsz)]
        for idx in range(self.bsz):
            question = self.tokenizer.decode(gen_batch.batch["question"][idx], skip_special_tokens=True)
            if "file_name" in gen_batch.non_tensor_batch.keys():
                # for gaia, we need additional file
                file_name = gen_batch.non_tensor_batch["file_name"]
            else:
                # for asearcher,arpo there is no file_name
                file_name = None
            self.prefix_prompt[idx] = self._build_system_prompt(question, file_name)
        self.is_final = torch.zeros(self.bsz, dtype=torch.bool)
        self.faiss_index = faiss.IndexFlatL2(1024)
        embedding = Qwen3Embedding()
        self.vectorstore = FAISS(embedding, self.faiss_index, docstore= InMemoryDocstore(),
                index_to_docstore_id={})
        self.agent_step = [0]*self.bsz
        self.context_step = torch.zeros(self.bsz, dtype=torch.long)
        self.temp_memory = [[] for _ in range(self.bsz)]
        self.short_memory = [""]*self.bsz
        self.query = ["default query, change it!"]*self.bsz
        self.stream_input_buffer = [[] for _ in range(self.bsz)]
        self.query_times = [0]*self.bsz
        self.recent_action = [[] for _ in range(self.bsz)]
        self.tool_text = [""]*self.bsz


    async def initialize(self) -> bool:
        """
        初始化测试，连接MCPManager并加载工具列表

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("初始化MCPManager客户端...")
            # 手动重试3次
            if not await self.mcp_manager.initialize():
                if not await self.mcp_manager.initialize():
                    if not await self.mcp_manager.initialize():
                        logger.error("MCPManager客户端初始化失败")
                        return False
            
            # 获取所有工具
            self.all_tools = self.mcp_manager.openai_tools
            
            logger.info(f"从MCPManager获取到 {len(self.all_tools)} 个工具")
            print(self.all_tools)
            
            # 按服务器组织工具
            servers = await self.mcp_manager.list_servers()
            self.tool_to_server_map = {}  # 添加工具到服务器的映射
            
            for server in servers:
                server_tools = await self.mcp_manager.get_server_tools(server)
                self.tools_by_server[server] = server_tools
                logger.info(f"服务器 '{server}' 上有 {len(self.tools_by_server[server])} 个工具")
                
                # 为每个工具创建到服务器的映射
                for tool in server_tools:
                    if "function" in tool:
                        tool_name = tool["function"].get("name", "unknown")
                        self.tool_to_server_map[tool_name] = server
                
            return True
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            traceback.print_exc()
            return False
        
    # TODO: 设计context的逻辑fix代码，包括交互历史如何给模型，工具如何给模型
    # 暂时方案：依旧按照单轮长prompt的逻辑，即action应当form的message的结构为：
    # [{"role": "system", "content": ...}, {"role": "user", "content": "query"}, {"role": "user", "content": "compressed memory"}]
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        self.active_mask = ~self.is_final # final -> not active
        active_indices = torch.where(self.active_mask)[0]
        self.messages = []
        for idx in active_indices:
            memory_text = ""
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20, "filter": {"trajectory_id": idx},"fetch_k": 5000})
            memory = retriever.invoke(self.query[idx])
            self.temp_memory[idx] = memory
            memory_text += "\n".join(f"Memory {i+1}: {doc.page_content}" for i, doc in enumerate(memory)) if memory else self.NO_MEMORY_TOKENS
            del retriever
            short_memory, long_memory = self.truncate_by_tokens(self.short_memory[idx], self.config.max_memorization_length, memory_text)
            message = self.prefix_prompt[idx].copy()
            message.append({"role": "user", "content": TEMPLATE_MEMORY.format(
                query=self.query[idx],
                short_memory=short_memory,
                query_times=self.query_times[idx], 
                long_memory=long_memory)})
            # ---- 添加最近一轮的交互数据 ----
            if self.agent_step[idx] < self.config.max_interactions -1:
                message.extend(self.recent_action[idx])
                message = self.tokenizer.apply_chat_template(message, tools=self.all_tools, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
                self.messages.append(message)
            else:
                message.append({"role": "user", "content": "This is the last step, all tools are disabled, give the final answer."})
                message = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")[0]
                self.messages.append(message)
        meta_info = {'input_pad_to': self.max_input_length,
                    'pad_to': self.config.gen_pad_to,
                    'generation_kwargs': {
                    'max_tokens': self.config.gen_max_tokens_final_response,
                    'n': 1 # note that we have already repeat n times in ray_trainer
                    }}
        logger.info(f'MemoryAgent.action() done')
        return self.messages, meta_info

    async def tool_call_async(self, texts, true_indices, action_rate_metric):
        async def async_wrapper(idx, real_idx):
            text = self.tokenizer.decode(texts[idx], skip_special_tokens=True)
            
            # 创建临时metric
            temp_metric = {k: 0 for k in action_rate_metric if k != "total"}
            self.parse_action(real_idx, text, temp_metric)
            
            # --- 异步调用 ---
            tool_call_text, tool_message = await self._parse_tool_call(text, temp_metric)
            
            # --- 将工具结果update到recent_action中 ---
            if tool_message:
                # 检查context buffer和context step是否置空，如果没有则在此置空
                self.context_step[real_idx] = 0
                self.stream_input_buffer[real_idx] = []
                # 如果本轮有工具调用，则按照<sliding window size直接放入recent_action，>sliding window size放入buffer中
                tool_content = tool_message["content"]
                if isinstance(tool_content, str):
                    recent_round = [{"role": "user", "content": f"Previous tool call:{tool_call_text}\n tool response: {tool_content}"}]
                elif isinstance(tool_content, dict):
                    recent_round = [{"role": "user", "content": f"Previous tool call:{tool_call_text}\n tool response: {json.dumps(tool_content)}"}]
                else:
                    logger.error(f"""tool call response got unexpected type: {type(tool_content)}""")
                token_length = len(self.tokenizer.apply_chat_template(recent_round))
                logger.info(f"Current Tool Response Length:{token_length}")
                if token_length <= self.config.max_sliding_window_length:
                    # 直接放入self.recent action中
                    self.recent_action[real_idx] = recent_round
                else:
                    context = json.dumps(tool_content,ensure_ascii=False)
                    self.stream_input_buffer[real_idx] = self.chunk_context(context)
                    total = len(self.stream_input_buffer[real_idx])
                    logger.info(f"stream input activated, chunked into {total} chunks.")
                    self.tool_text[real_idx] = tool_call_text
                    # self.recent_action[real_idx] = [{"role": "assistant", "content": tool_call_text}, {"role": "tool", "content": CHUNK_CONTEXT_TEMPLATE.format(chunk = self.stream_input_buffer[real_idx][0], current_idx=1, total = total)}]
                    self.recent_action[real_idx] = [{"role": "tool", "content": CHUNK_CONTEXT_TEMPLATE.format(tool_call_text=tool_call_text, chunk = self.stream_input_buffer[real_idx][0], current_idx=1, total = total)}]
            else:
                # 如果没有工具调用，看是否要silding window size滑动一步
                # 如果已经没有context了
                if self.context_step[real_idx] >= len(self.stream_input_buffer[real_idx]):
                    self.recent_action[real_idx] = [{"role": "user", "content": f"previous response: {text}"}]
                else:
                    logger.info(f"stream input activated, reading {self.context_step[real_idx]+1} chunks, total:{len(self.stream_input_buffer[real_idx])}.")
                    total = len(self.stream_input_buffer[real_idx])
                    # self.recent_action[real_idx] = [{"role": "assistant", "content": self.tool_text[real_idx]}, {"role": "tool", "content": CHUNK_CONTEXT_TEMPLATE.format(chunk = self.stream_input_buffer[real_idx][self.context_step[real_idx]], current_idx=self.context_step[real_idx]+1, total = total)}]
                    self.recent_action[real_idx] = [{"role": "tool", "content": CHUNK_CONTEXT_TEMPLATE.format(tool_call_text=self.tool_text[real_idx],chunk = self.stream_input_buffer[real_idx][self.context_step[real_idx]], current_idx=self.context_step[real_idx]+1, total = total)}]
            
            # --- agent多走一步 ---
            self.agent_step[real_idx] += 1
            if self.agent_step[real_idx] > self.config.max_interactions:
                self.is_final[real_idx] = True
            
            return temp_metric
        tasks = [async_wrapper(idx, real_idx) for idx, real_idx in enumerate(true_indices)]
        return await asyncio.gather(*tasks)
        

    
    # TODO: context如何更新，is_final如何判定，memory如何更新
    @override
    def update(self, gen_output: DataProto) -> DataProto:
        action_rate_metric = {"action_rate/update_query": 0, "action_rate/add_memory": 0, "action_rate/modify_memory": 0, "action_rate/delete_memory": 0, "action_rate/tool_call_success": 0, "action_rate/tool_call_fail": 0, "total": gen_output.batch['responses'].shape[0]}
        if not torch.all(self.is_final):
            true_indices = torch.where(self.active_mask)[0].tolist()
            texts = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            metric_results = asyncio.run(self.tool_call_async(texts, true_indices, action_rate_metric))
            # 汇总metric结果
            for metric in metric_results:
                for key in metric:
                    if key in action_rate_metric:
                        action_rate_metric[key] += metric[key]
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
    
    # TODO: 关闭已有资源
    @override
    def end(self):
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        self.dump_vectorstore("/nfsdata/huoyupeng/Memory_Agent/dataset.txt")
        del self.vectorstore
        del self.faiss_index
        del self.query
        del self.temp_memory
        del self.prefix_prompt
        del self.stream_input_buffer
        del self.recent_action
        del self.tool_text
        return final_mask, sample_index, []

    def _build_system_prompt(
        self,
        query: str,
        file_name: Optional[str] = None):
        # --- System Prompt 构建 (保持不变) ---
        system_message = SYSTEM_PROMPT_QWEN
        if file_name:
            full_path = os.path.join('/app/data/gaia_validation', file_name).replace('\\', '/')
            system_message += "\n\n--- Associated File ---\n"
            system_message += f"A file is associated with this task.\n"
            system_message += f"The absolute path to this file for your tools is: {full_path}\n"
        prefix_prompt = [
                {"role": "system", "content": system_message}
            ]
            
        prefix_prompt.append({"role": "user", "content": f"Your task is to answer the user's question: {query}"})
        tokend_prompt = self.tokenizer.apply_chat_template(prefix_prompt)
        assert len(tokend_prompt) <= self.config.max_prompt_length
        return prefix_prompt
    
    async def _parse_tool_call(self, response : str, metric: dict):
        if '<tool_call>' in response:
            tool_start_tag = '<tool_call>'
            tool_end_tag = '</tool_call>'
            start_index = response.find(tool_start_tag)
            end_index = response.rfind(tool_end_tag)
            if end_index > start_index:
                json_str_to_parse = response[start_index + len(tool_start_tag):end_index].strip()
                response = response[:start_index].strip()
                try:
                    json_tools = json.loads(json_str_to_parse)
                    tool_name = json_tools.get("name", "")
                    server_id = self.tool_to_server_map.get(tool_name)
                    full_tool_name = f"{server_id}.{tool_name}" if "." not in tool_name and server_id else tool_name
                    arguments_input = json_tools.get("arguments", "{}")
                    arguments = self._try_parse_tool_call_arguments(arguments_input)
                    if tool_name == "search":
                        # default 10 results
                        if "num_results" not in arguments:
                            arguments["num_results"] = 10
                    tool_result = await self.mcp_manager.call_tool(full_tool_name, arguments)
                    if tool_result["status"] == "error":
                        metric["action_rate/tool_call_fail"] += 1
                        error_msg = tool_result.get("content", {}).get("error", "未知错误")
                        error_detail = tool_result.get("content", {}).get("detail", "")
                        error_traceback = tool_result.get("content", {}).get("traceback", "")
                        error_content = {"error": error_msg, "server_id": server_id}
                        if error_detail: error_content["detail"] = error_detail
                        if error_traceback: error_content["traceback"] = error_traceback
                        error_content_str = json.dumps(error_content,ensure_ascii=False)
                        tool_message = {
                            "role": "tool", 
                            "content": error_content_str
                        }
                    else:
                        metric["action_rate/tool_call_success"] += 1
                        tool_content = tool_result.get("content", {})
                        tool_message = {
                                "role": "tool", 
                                "content": tool_content
                            }
                    return json_str_to_parse, tool_message
                except Exception as e:
                    logger.error(f"工具字符串解析错误:{e}", exc_info=True)
                    logger.error(f"原始的工具字符串：{json_str_to_parse}")
                    return "", {"role": "tool", "content": f"工具字符串解析错误{e}"}
            else:
                return "", {}
        else:
            return "", {}
        

    def _try_parse_tool_call_arguments(self, arguments_input: Union[str, Dict]) -> Dict[str, Any]:
        
        # 如果输入本身就已经是字典，直接返回
        if isinstance(arguments_input, dict):
            return arguments_input
        
        # 如果输入是字符串，尝试解析
        if isinstance(arguments_input, str):
            arguments_str = arguments_input.strip()
            if not arguments_str:
                return {}
            
            # 尝试标准JSON解析
            try:
                return json.loads(arguments_str)
            except json.JSONDecodeError:
                
                # 如果标准解析失败，检查是不是因为缺少了结尾的 '}'
                if arguments_str.startswith('{') and not arguments_str.endswith('}'):
                    logger.info("检测到JSON可能缺失右大括号，尝试自动修复...")
                    healed_str = arguments_str + '}'
                    try:
                        # 再次尝试解析修复后的字符串
                        return json.loads(healed_str)
                    except json.JSONDecodeError:
                        # 如果修复后仍然失败，就放弃修复，让后续逻辑处理
                        logger.warning("自动修复后解析依然失败，继续尝试其他解析方法。")
                

                # 尝试使用json5进行更宽松的解析（例如，处理单引号或末尾逗号）
                try:
                    import json5
                    return json5.loads(arguments_str)
                except Exception as e:
                    # 如果所有解析都失败，返回一个包含原始字符串的默认结构
                    logger.warning(f"无法将参数 '{arguments_str}' 解析为JSON，将作为单一查询处理。错误: {e}")
                    return {"query": arguments_str}
        
        # 对于其他意外类型，返回空字典
        return {}
    
    def parse_action(self, idx: int, text: str, action_rate_metric: Dict[str, int]):
        """
        支持在一次响应里出现多个 <tag>…</tag> 动作。
        按照“先匹配先执行”原则处理；如无任何合法标签则记 not_follow。
        """
        # 1. 提取所有成对标签区间
        try:
            tags = self._extract_paired_tags(text)
            if not tags:
                self.context_step[idx] += 1
                return
            move_on = True
            for tag_name, content in tags:
                tag_name = tag_name.lower()
                if tag_name == 'add_memory':
                    self._handle_add(idx, content.strip(), action_rate_metric)
                elif tag_name == 'update_memory':
                    self._handle_update(idx, content.strip(), action_rate_metric)
                elif tag_name == 'modify_memory':
                    self._handle_modify(idx, content.strip(), action_rate_metric)
                elif tag_name == 'delete_memory':
                    self._handle_delete(idx, content.strip(), action_rate_metric)
                elif tag_name == 'update_query':
                    self._handle_query(idx, content.strip(), action_rate_metric)
                    move_on = False
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

    def _handle_update(self, idx: int, content: str, metric: Dict[str, int]):
        self.short_memory[idx] = content
        self.query_times[idx] = 0
        
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

    def _handle_delete(self, idx: int, content: str, metric: Dict[str, int]):
        m = re.match(r'Memory\s+(\d+)', content, flags=re.S)
        if not m:
            metric['action_rate/not_follow'] += 1
            return
        doc_idx = int(m.group(1)) - 1
        if doc_idx < len(self.temp_memory[idx]):
            old_content = self.temp_memory[idx][doc_idx].page_content
            self.modify_document(idx, old_content, "")
        metric['action_rate/delete_memory'] += 1
        self.query_times[idx] = 0
        
    def _handle_query(self, idx: int, query: str, metric: Dict[str, int]):
        if self.query_times[idx] > 2:
            self.is_final[idx] = True
            return
        if query: 
            self.query[idx] = query
        metric['action_rate/update_query'] += 1
        self.query_times[idx] += 1

    def chunk_context(self, ctx_str):
        """
        带重叠的分块，每个chunk包含前后上下文
        """
        chunk_size = self.config.chunk_size
        overlap = 100  # 重叠的token数
        ctx_ids = self.tokenizer.encode(ctx_str,add_special_tokens=False)
        chunks = []
        start = 0
        
        while start < len(ctx_ids):
            # 计算chunk的结束位置
            end = min(start + chunk_size + 2*overlap, len(ctx_ids))
            
            # 获取当前chunk的token IDs
            chunk_ids = ctx_ids[start:end]
            
            # 解码回字符串
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # 移动到下一个chunk，考虑重叠
            start += chunk_size - overlap
            
            # 避免无限循环
            if start >= len(ctx_ids):
                break
        
        return chunks

    @staticmethod
    def _extract_paired_tags(text: str):
        """
        返回 [(tag_name, inner_content), ...]，按出现顺序。
        只做最外层匹配，不处理嵌套。
        """
        pattern = re.compile(r'<(\w+)[^>]*>(.*?)</\1>', flags=re.S | re.I)
        return pattern.findall(text)
    
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

    def truncate_by_tokens(self, text: str, max_tokens: int, addition_text: str= "") -> str:
        """
        Truncate a string by token count using a given tokenizer.

        Args:
            text (str): input text
            max_tokens (int): maximum number of tokens allowed
            tokenizer: tokenizer with encode/decode methods

        Returns:
            str: truncated text
        """
        # Encode into tokens
        tokens = self.tokenizer.encode(text)
        addition_tokens = self.tokenizer.encode(addition_text)

        #分类讨论
        # If already within limit
        if len(tokens)+len(addition_tokens) <= max_tokens:
            return text, addition_text
        elif len(tokens) >= max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens)
            return truncated_text, ""
        else:
            truncated_addition_tokens = addition_tokens[len(tokens):max_tokens]
            truncated_addition_text = self.tokenizer.decode(truncated_addition_tokens)
            return text, truncated_addition_text

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
        if new_content:
            self._write_memory_to_vectorstore(traj_id, new_content)

    def log_step(self, gen_output):
        """
        Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=40000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.agent_step[:10] if not torch.all(self.is_final) else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] Agent STEP {step}{'='*30}")

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


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=GAIAMemoryConfig, dataset_cls=MemoryDataset, agent_cls=GAIAMemoryAgent)
