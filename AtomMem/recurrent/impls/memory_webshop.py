import re
import os
import pdb
import yaml
import requests
import logging
from bs4 import BeautifulSoup
from bs4.element import Comment
from pathlib import Path
from difflib import get_close_matches
from urllib.parse import quote_plus
from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Union, Dict
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F
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

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.max_memorization_length + self.max_sliding_window_length
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
        data_config.max_prompt_length=recurrent_config.max_prompt_length
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
        
        row_dict["instruction"] = F.pad(instruction, (0, self.max_prompt_length- instruction.size(0)), value=self.tokenizer.pad_token_id)
        
        index = row_dict.get("extra_info", {}).get("index", "train_0")
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["instruction"], ["index"]


def clean_str(p):
    return p.encode('latin-1', errors='ignore').decode('latin-1')


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
            element.parent.name not in ignore and not isinstance(element, Comment)
    )

class PageNumberError(Exception):
    pass

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

TEMPLATE = """You are a web-agent equipped with a **working memory** that can be explicitly updated.  
Your task: **gather information about the user’s problem by interacting with the website** and **managing your own memory**. 
You can perform searches, click on item pages, judge whether the items meet the user's needs based on their descriptions, and compare several items to find better options. You should record in your memory the items you have seen and whether they meet the requirements, for example, by adding a "(selected)" identifier after the temporarily optimal solution. Remember to record item ID. Keep gathering new information—even when you think you’ve found the best product, something better may still exist.

Three kinds of memory actions are available: <update_query>  <add_memory> <modify_memory>
<update_query>: The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query. DO NOT repeatly update query, if you don’t have the desired memory, it means the entry does not exist in the knowledge base. 

<add_memory>: creates a new entry in the memory. You do not need to repeatly add the memory shown to you or enter index of the memory. 

<modify memory> updates the existing entry, You need to enter the memory index "Memory i:" to specify which memory to modify.

The user request:
{prompt}

Current query to retrieve memory:
{query}

Current memory:
{memory}

Current website:
<website>
{website}
</website>

Available actions(Your output format must exactly match one of these):
{available_actions}

Your recent action history(Be careful not to repeat similar actions):
{history}

You have a maxium of {max_chunk} steps, you have performed {agent_step} step.

Output format:
- no-thinking, action only.  
- **At most one** website action inside `<web></web>` in the end of response. If you choose to update query, you can take no action on current website.
- Any number of memory actions with `<></>`, such as <update_query></update_query> or <add_memory></update_query>.

Example of actions:
<update_query>The dance partner of Yulia Zagoruychenko.</update_query><web>search[Youlia Zagoruychenko]</web>

<add_memory>B09H7SWQ: queen sized bed with blue and pink(selected)</add_memory><web>click[Next >]</web>

<modify_memory>Memory 1: Yulia Zagoruychenko's dance partner is not Riccardo. He is Emily's dance partner, I need to find another guy.</modify_memory><web>click[Back to Search]</web>
"""

FINAL_TEMPLATE = """You are a web-agent equipped with a **working memory** that can be explicitly updated.  
Your task: **solve the user’s problem by interacting with the website** and **managing your own memory**. 

You have collected information. Now based on your memory, select and answer with the most appropriate item. You must select an item, even if it does not meets all the requirements. Example:<web>click[B09H7SWQ]</web>. Enter the item page, choose the type you want to buy, and click "Buy Now" to complete the task.

Three kinds of memory actions are available: <update_query>  <add_memory> <modify_memory>
<update_query>: The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query. DO NOT repeatly update query, if you don’t have the desired memory, it means the entry does not exist in the knowledge base. 

<add_memory>: creates a new entry in the memory. You do not need to repeatly add the memory shown to you or enter index of the memory. 

<modify memory> updates the existing entry, You need to enter the memory index "Memory i:" to specify which memory to modify.

The user request:
{prompt}

Current query to retrieve memory:
{query}

Current memory:
{memory}

Current website:
<website>
{website}
</website>

Available actions(Your output format must exactly match one of these):
{available_actions}

Your recent action history(Be careful not to repeat similar actions):
{history}

You have a maxium of 10 steps, you have performed {agent_step} step. Please note that exceeding the maximum number of steps will result in task failure.

Output format:
- no-thinking, action only.    
- **At most one** website action inside `<web></web>` in the end of response. If you choose to update query, you can take no action on current website.
- Any number of memory actions with `<></>`, such as <update_query></update_query> or <add_memory></update_query>.

Example of actions:
<update_query>The dance partner of Yulia Zagoruychenko.</update_query>.

<web>click[B09H7SWQ]</web>
"""


class Webshop(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig, web_url="http://127.0.0.1:3000"):
        self.config = config
        self.tokenizer = tokenizer
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + 8192
        logger.info(f'\n[RECURRENT] max_input_length: = {self.max_input_length}\n')
        logger.info(f"chunk_size: {self.config.chunk_size}")
        self.NO_MEMORY_TOKENS = "No previous memory"
        self.web_url = web_url
        
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        self.bsz = len(gen_batch.non_tensor_batch["index"])
        self.final_turn = False
        self.is_final = torch.zeros(self.bsz, dtype=torch.bool)
        self.faiss_index = faiss.IndexFlatL2(1024)
        self.vectorstore = FAISS(self.call_remote_embedding, self.faiss_index, docstore= InMemoryDocstore(),
                index_to_docstore_id={})
        self.agent_step = [0]*self.bsz
        # temp_memory is used for modify
        self.temp_memory = [[]]*self.bsz
        self.query = ["No query"]*self.bsz
        self.query_times = [0]*self.bsz
        self.history = [""]*self.bsz
        self.sessions = [{}]*self.bsz
        self.website = [""]*self.bsz
        self.reward = [0.0]*self.bsz
        self.prompt = [""]*self.bsz
        self.is_buy = [False]*self.bsz
        for i in range(self.bsz):
            session = gen_batch.non_tensor_batch["index"][i]
            self.sessions[i] = {'session': session, 'page_type': 'init'}
            observation, info, done = self.step(i, "reset[]")
            assert "instruction" in info
            self.prompt[i] = info["instruction"]
            self.website[i] = observation

    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        self.active_mask = ~self.is_final # final -> not active
        self.messages = []
        active_indices = torch.where(self.active_mask)[0]
        for idx in active_indices:
            # build the message, we need problem, query, memory, website(observation) and history
            # here we get memory, query, problem, the website will be collected in update phase.
            if self.query[idx] != "No query":
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"trajectory_id": idx},"fetch_k": 5000})
                memory = retriever.invoke(self.query[idx])
                self.temp_memory[idx] = memory
                memory_text = "\n".join(f"Memory {i+1}: {doc.page_content}" for i, doc in enumerate(memory))[:self.config.max_memorization_length] if memory else self.NO_MEMORY_TOKENS
                del retriever
            else:
                memory_text = self.NO_MEMORY_TOKENS
            prompt = self.prompt[idx]
            available_actions = self.get_action_space(idx, is_buy=self.is_buy[idx])
            if self.agent_step[idx] < self.config.max_chunks - 10:
                message = [{"role": "user", "content": TEMPLATE.format(prompt = prompt, query = self.query[idx], memory=memory_text, history=self.history[idx][-self.config.max_sliding_window_length:], website=self.website[idx], available_actions=available_actions, max_chunk=self.config.max_chunks - 10, agent_step=self.agent_step[idx])}]
            else:
                message = [{"role": "user", "content": FINAL_TEMPLATE.format(prompt = prompt, query = self.query[idx], memory=memory_text, history=self.history[idx][-self.config.max_sliding_window_length:], website=self.website[idx], available_actions=available_actions, agent_step=10 + self.agent_step[idx] - self.config.max_chunks)}]
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
        action_rate_metric = {"action_rate/do_nothing":0,"action_rate/not_follow": 0, "action_rate/update_query": 0, "action_rate/add_memory": 0, "action_rate/modify_memory": 0, "total": gen_output.batch['responses'].shape[0]}
        if not torch.all(self.is_final):
            true_indices = torch.where(self.active_mask)[0].tolist()
            texts = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            for idx, real_idx in enumerate(true_indices):
                text = self.tokenizer.decode(texts[idx], skip_special_tokens=True)
                self.history[real_idx] += "\n-------------\n" + text
                # manipulating memory
                self.parse_memory_action(real_idx, text, action_rate_metric)
                self.agent_step[real_idx] += 1
                # manipulation website
                action = self.parse_website_action(text)
                if self.agent_step[real_idx] == self.config.max_chunks - 10:
                    # buy_1用于在切换的那一步忽略action，随后切换到buy_2
                    self.sessions[real_idx] = {'session': self.sessions[real_idx]['session'], 'page_type':'buy_1'}
                    self.is_buy[real_idx] = True
                    self.history[real_idx] = ""
                observation, info, done = self.step(real_idx, action, is_buy=self.is_buy[real_idx])
                if observation != 'Preserve':
                    self.website[real_idx] = observation
                self.is_final[real_idx] = done
                self.reward[real_idx] = info.get("reward", 0.0)
                # 超出最大步数直接结束，防止无限循环
                if self.agent_step[real_idx] > self.config.max_chunks:
                    self.is_final[real_idx] = True
                
        sample_index = torch.arange(self.bsz, dtype=torch.long)[self.active_mask] # map active sample to original batch
        self.sample_index_list.append(sample_index)
        final_mask = self.is_final[self.active_mask]
        self.final_mask_list.append(final_mask)
        gen_output.meta_info["action_rate_metric"] = action_rate_metric
        # self.log_step(gen_output)
        return gen_output
    
    @override
    def done(self):
        return torch.all(self.is_final)
    
    @override
    def end(self):
        del self.gen_batch
        del self.meta_info
        del self.messages
        self.dump_vectorstore("/home/test/test03/huoyupeng/MemAgent/MemAgent-main/dataset.txt")
        del self.vectorstore
        del self.faiss_index
        del self.query
        del self.temp_memory
        del self.history
        del self.website
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index, self.reward
    
    def parse_memory_action(self, idx: int, text: str, action_rate_metric: Dict[str, int]):
        """
        支持在一次响应里出现多个 <tag>…</tag> 动作。
        按照“先匹配先执行”原则处理；如无任何合法标签则记 not_follow。
        """
        # 1. 提取所有成对标签区间
        do_something = False
        try:
            tags = self._extract_paired_tags(text)
            if not tags:
                action_rate_metric['action_rate/not_follow'] += 1
                return
            for tag_name, content in tags:
                tag_name = tag_name.lower()
                if tag_name == 'add_memory':
                    self._handle_add(idx, content.strip(), action_rate_metric)
                    do_something = True
                elif tag_name == 'modify_memory':
                    self._handle_modify(idx, content.strip(), action_rate_metric)
                    do_something = True
                elif tag_name == 'update_query':
                    self._handle_query(idx, content.strip(), action_rate_metric)
                    do_something = True
            if not do_something:
                    action_rate_metric['action_rate/do_nothing'] += 1
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
        
    @staticmethod
    def _extract_paired_tags(text: str):
        """
        返回 [(tag_name, inner_content), ...]，按出现顺序。
        只做最外层匹配，不处理嵌套。
        """
        pattern = re.compile(r'<(\w+)[^>]*>(.*?)</\1>', flags=re.S | re.I)
        return pattern.findall(text)
    
    def parse_website_action(self, text):
        pattern = re.compile(r'<web[^>]*>(.*?)</web>', flags=re.S | re.I)
        tags = pattern.findall(text)
        pattern_2 = re.compile(r'<update_query[^>]*>(.*?)</update_query>', flags=re.S | re.I)
        tag_2 = pattern_2.findall(text)
        if len(tags) > 1:
            return None
        elif len(tags) == 0:
            if len(tag_2) == 1:
                return "update_query"
            else:
                return None
        else:
            action = tags[0]
            return action
            

    def update_item_actions(self, idx):
        basic_action_dic = self.basic_action_dic
        # get option_types
        option_types = self.sessions[idx].get('option_types', [])
        # generate click[option_type]
        option_actions = [f"click[{option_type}]" for option_type in option_types]
        # update basic_action_dic
        basic_action_dic["item"] = self.basic_action_dic["item"] + option_actions

        return basic_action_dic["item"]

    def update_search_actions(self, idx):
        basic_action_dic = self.basic_action_dic
        asins = self.sessions[idx].get('asins', [])
        actions = [f"click[{asin}]" for asin in asins]
        basic_action_dic["search"] = self.basic_action_dic["search"] + actions

        return basic_action_dic["search"]

    def get_action_space(self, idx, is_buy=False):
        self.basic_action_dic = {
            "init": ["search[]"],
            "search": ["click[Next >]", "click[< Prev]", "click[Back to Search]"],
            "item": ["click[Buy Now]", "click[< Prev]", "click[Description]", "click[Features]", "click[Reviews]",
                     "click[Attributes]", "click[Back to Search]"],
            "item_sub": ["click[Back to Search]", "click[< Prev]"],
        }
        page_type = self.sessions[idx]['page_type']
        if page_type == 'buy_2':
            return ["click[]"]
        elif page_type == 'item':
            valid_actions = self.update_item_actions(idx)
            if not is_buy:
                valid_actions.remove("click[Buy Now]")
            if is_buy:
                valid_actions.remove("click[Back to Search]")
                valid_actions.remove("click[< Prev]")
        elif page_type == 'search':
            valid_actions = self.update_search_actions(idx)
        else:
            valid_actions = self.basic_action_dic[page_type]
            if is_buy:
                valid_actions.remove("click[Back to Search]")
        return valid_actions

            

    def step(self, idx, action, is_buy=False):
        done = False
        # observation_ represents the action that won't update the website, but should be added in the history.
        observation_ = None
        reward = 0.0
        # modify the page type
        session = self.sessions[idx]['session']
        is_error = 0
        # 检查动作
        if self.sessions[idx]['page_type'] == 'buy_2':
            if not (isinstance(action, str) and action.startswith('click[')):
                observation = 'Preserve'
                info = {"reward": 0.0}
                self.history[idx] += 'You can only click on this page.'
                return observation, info, done
        # 初始页面
        if action == 'reset[]':
            pass
        # 转换页面
        elif self.sessions[idx]['page_type'] == 'buy_1':
            self.sessions[idx]['page_type'] = 'buy_2'
            observation = "WEB PAGE: {\nDetermine the item to buy based on your memory!\n}"
            info = {"reward": 0.0}
            return observation, info, done
        # 动作解析
        elif isinstance(action, str) and action.startswith('search['):
            if self.sessions[idx]['page_type'] == 'init':
                query = action[7:-1]
                self.sessions[idx] = {'session': session, 'page_type': 'search',
                                          'query_string': query, 'page_num': 1}
            else:
                observation_ = 'There is no [Search] button.'
        elif isinstance(action, str) and action.startswith('click['):
            button = action[6:-1]
            if button == 'Buy Now':
                try:
                    assert self.sessions[idx]['page_type'] == 'item' and is_buy # 只有在购买阶段才能买东西
                    self.sessions[idx]['page_type'] = 'end'
                    done = True
                except:
                    is_error = 1
            elif button == 'Back to Search':
                try:
                    assert self.sessions[idx]['page_type'] in ['search', 'item_sub', 'item'] and not is_buy # 尽管我们不会在购买阶段给我们提供Bach to Search，仍然防止其误触
                    self.sessions[idx] = {'session': session, 'page_type': 'init'}
                except:
                    is_error = 1
            elif button == 'Next >':
                try:
                    assert self.sessions[idx]['page_type'] == 'search'
                    if self.sessions[idx]['page_num'] > 15:
                        raise PageNumberError
                    self.sessions[idx]['page_num'] += 1
                except:
                    is_error = 1
            elif button == '< Prev':
                try:
                    assert self.sessions[idx]['page_type'] in ['search', 'item_sub', 'item']
                    if self.sessions[idx]['page_type'] == 'search':
                        if self.sessions[idx]['page_num'] == 1:
                            raise PageNumberError
                        self.sessions[idx]['page_num'] -= 1
                    elif self.sessions[idx]['page_type'] == 'item_sub':
                        self.sessions[idx]['page_type'] = 'item'
                    elif self.sessions[idx]['page_type'] == 'item':
                        assert not is_buy # item界面在购买阶段不会有[< Prev]按钮
                        self.sessions[idx]['page_type'] = 'search'
                        self.sessions[idx]['options'] = {}
                except:
                    is_error = 1
            elif button in ACTION_TO_TEMPLATE:
                try:
                    assert self.sessions[idx]['page_type'] == 'item'
                    self.sessions[idx]['page_type'] = 'item_sub'
                    self.sessions[idx]['subpage'] = button
                except:
                    is_error = 1
            else:
                try:
                    if self.sessions[idx]['page_type'] == 'search':
                        assert button in self.sessions[idx].get('asins', [])  # asin必须在当前页面上
                        self.sessions[idx]['page_type'] = 'item'
                        self.sessions[idx]['asin'] = button
                    elif self.sessions[idx]['page_type'] == 'buy_2': # asin不必在当前页面上
                        pattern = re.compile(r'^[A-Z0-9]+$')
                        assert pattern.match(button)
                        self.sessions[idx]['page_type'] = 'item'
                        self.sessions[idx]['asin'] = button
                        self.sessions[idx]['query_string'] = 'abc' # there must be a query string, i don't know why
                    elif self.sessions[idx]['page_type'] == 'item':
                        assert 'option_types' in self.sessions[idx]
                        assert button in self.sessions[idx]['option_types'], (
                            button, self.sessions[idx]['option_types'])  # must be options
                        option_type = self.sessions[idx]['option_types'][button]
                        if not 'options' in self.sessions[idx]:
                            self.sessions[idx]['options'] = {}
                        self.sessions[idx]['options'][option_type] = button
                        observation_ = f'You have clicked {button}.'
                except:
                    is_error = 1
        elif action == 'update_query':
            observation = 'Preserve'
            info = {"reward": 0.0}
            self.history[idx] += 'query updated.'
            return observation, info, done
        else:
            is_error = 1
        if is_error:
            observation_ = 'Incorrect action format. Please use the correct action format following:\n' \
                          'Available Actions:\n\nclick[something]: Engage with specific buttons or links.\n' \
                          'search[something]: Seek specific data on the website. Use this only if a [Search] button ' \
                          'appears in the observation.\n' \
                          'Note: If you wish to search and there is no [Search] button, click the [Back to Search] ' \
                          'button instead. '
            self.history[idx] = self.history[idx] + '\n' + observation_
            # 保持前一轮打开的网站
            observation = 'Preserve'
            info = {"reward": 0.0}
            return observation, info, done
        try:
            observation, info = self.webshop_text(**self.sessions[idx])
        except:
            observation = 'Preserve'
            info = {"reward": 0.0}
            return observation, info, done
        if observation_:
            self.history[idx] = self.history[idx] + '\n' + observation_
        self.sessions[idx].update(info)
        pattern = re.compile(r'Instruction:\s*(.*)\s*\[')
        match = pattern.search(observation)
        if match:
            observation = pattern.sub('', observation).strip()
        # 在特定环节删除对应的按钮
        if not is_buy and self.sessions[idx]['page_type'] == 'item':
            observation = observation.replace("[Buy Now]", "")
        if is_buy and self.sessions[idx]['page_type'] == 'item':
            observation = observation.replace("[< Prev]", "")
            observation = observation.replace("[Back to Search]", "")
        if is_buy and self.sessions[idx]['page_type'] == 'item_sub':
            observation = observation.replace("[Back to Search]", "")
        observation = "WEB PAGE: {" + observation + "}"
        return observation, info, done
    
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

    def webshop_text(self, session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
        if page_type == 'init':
            url = (
                f'{self.web_url}/{session}'
            )
        if page_type == 'search':
            query_string = quote_plus(query_string)
            url = (
                f'{self.web_url}/search_results/{session}/'
                f'{query_string}/{page_num}'
            )
        elif page_type == 'item':
            query_string = quote_plus(query_string)
            options = {k: quote_plus(v) for k, v in options.items()}
            url = (
                f'{self.web_url}/item_page/{session}/'
                f'{asin}/{query_string}/{page_num}/{options}'
            )
        elif page_type == 'item_sub':
            query_string = quote_plus(query_string)
            options = {k: quote_plus(v) for k, v in options.items()}
            url = (
                f'{self.web_url}/item_sub_page/{session}/'
                f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
            )
        elif page_type == 'end':
            options = {k: quote_plus(v) for k, v in options.items()}
            url = (
                f'{self.web_url}/done/{session}/'
                f'{asin}/{options}'
            )
        # Mark request URL
        request_id = 'Resquest: ' + url
        headers = {'X-Request-ID': request_id}
        try:
            html = requests.get(url, headers=headers).text
            html_obj = BeautifulSoup(html, 'html.parser')
        except:
            observation = 'Perserve'
            info = {}
        texts = html_obj.findAll(text=True)
        visible_texts = list(filter(tag_visible, texts))
        if False:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            option_type = ''
            options = {}
            asins = []
            cnt = 0
            prod_cnt = 0
            just_prod = 0
            skip_counter = 0
            instruction = ""
            for i, t in enumerate(visible_texts):
                if skip_counter > 0:
                    skip_counter -= 1
                    continue  # progress score is invisible
                if t == '\n': continue
                if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
                if t.parent.name == 'button':  # button
                    processed_t = f'\n[{t}] '
                elif t.parent.name == 'label':  # options
                    if f"'{t}'" in url:
                        processed_t = f'[[{t}]]'
                    else:
                        processed_t = f'[{t}]'
                    options[str(t)] = option_type
                elif t.parent.get('class') == ["product-link"]:  # product asins
                    processed_t = f'\n[{t}] '
                    if prod_cnt >= 7:
                        processed_t = ''
                    prod_cnt += 1
                    asins.append(str(t))
                    just_prod = 0
                else:  # regular, unclickable text
                    processed_t = '\n' + str(t) + ' '
                    if cnt < 2 and page_type != 'init': processed_t = ''
                    option_type = str(t)
                    cnt += 1
                just_prod += 1
                observation += processed_t
            info = {}
            if options:
                info['option_types'] = options
            if asins:
                info['asins'] = asins
            if 'Your score (min 0.0, max 1.0)' in visible_texts:
                idx = visible_texts.index('Your score (min 0.0, max 1.0)')
                info['reward'] = float(visible_texts[idx + 1])
                observation = 'Result: [Success]' if float(visible_texts[idx + 1]) == 1.0 else 'Result: [False]'
            if "Instruction: " in visible_texts:
                idx = visible_texts.index('Instruction: ')
                info['instruction'] = visible_texts[idx + 1]
            return clean_str(observation), info
    
    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=9000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.agent_step[:10] if not torch.all(self.is_final) else "FINAL"
        logger.info(f"\n{'='*30}[RECURRENT] Agent STEP {step}{'='*30}")

        # Message and Response section
        if torch.any(self.active_mask):
            prt0 = gen_output.batch['prompts'][0]
            decoded_message = self.tokenizer.decode(prt0[prt0!=self.tokenizer.pad_token_id])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")
            
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=Webshop)
