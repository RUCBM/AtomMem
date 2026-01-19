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
from torch.nn import functional as F
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
from scienceworld import ScienceWorldEnv
from recurrent.envs.env_manager import SCIWorldMultiEnvManager

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
        
        row_dict["prompt"] = F.pad(instruction, (0, self.max_prompt_length- instruction.size(0)), value=self.tokenizer.pad_token_id)
        
        env_name = row_dict.get("extra_info", {}).get("env_name", "None")
        var = row_dict.get("extra_info", {}).get("var", "None")
        row_dict["env_name"] = env_name
        row_dict["var"] = var
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["prompt"], ["env_name", "var"]

TEMPLATE = """You are an agent in a virtual science school environment, tasked to interact with various elements.You are equipped with a **working memory** that can be explicitly updated.  
Your task: solve the user’s problem by interacting with the website** and **managing your own memory**. 

The user request:
{prompt}

Available memory actions: 1) update_query 2) add_memory 3) modify_memory
The system maintains a query for memory retrieval. You can modify it via query operations. You will not get any memory unless you give a query. DO NOT repeatly update query, if you don’t have the desired memory, it means the entry does not exist in the knowledge base. You do not need to repeat the memory shown to you, otherwise there will be a large number of duplicate entries. DO NOT REPEATLY UPDATE QUERY!

Add memory and modify_memory are used to update the memory bank. "add memory" creates a new entry, "modify memory" updates the existing entry, You need to enter the document number "Memory i:" to specify which document to replace. The document number is only need in modify.

Current query to retrieve memory:
{query}

Current memory:
{memory}

Following is the possible actions that interact with the environment (with short explanation):
- **Manipulation**:
- `open [OBJ]` / `close [OBJ]`: Interact with a container.
- `pick up [OBJ]`: Add an object to your inventory.
- `put down [OBJ]`: Remove an object from your inventory.
- `move [OBJ] to [OBJ]`: Transfer an object.
- `pour [OBJ] into [OBJ]`: Pour a substance.
- `dunk [OBJ] into [OBJ]`: Immerse a container in a liquid.
- `mix [OBJ]`: Chemically combine contents.

- **Inspection**:
- `look around`: Survey your surroundings.
- `look at [OBJ]`: Examine an object closely.
- `look in [OBJ]`: Peek inside a container. 
- `read [OBJ]`: Review written content. 

- **Device Operations**:
- `activate [OBJ]` / `deactivate [OBJ]`: Toggle a device.
- `use [OBJ] [on [OBJ]]`: Utilize a device or item.

- **Movement**:\n  - `go to [LOC]`: Relocate.

- **Miscellaneous**:
- `eat [OBJ]`: Consume an edible item.
- `flush [OBJ]`: Activate a flushing mechanism.
- `focus on [OBJ]`: Direct attention to a particular object.
- `wait [DURATION]`: Pause for a specified period.

- **Information**:\n  
- `task`: Recap your current objective.
- `inventory`: Display items you're carrying.
- `objects`: Display items around you.
Where:
- `[OBJ]`: Object
- `[LOC]`: Location
- `[DURATION]`: Specified time, which is a number(no time unit is needed).

You have a maxium of {max_chunk} steps, you have performed {agent_step} step. Please note that exceeding the maximum number of steps will lead to the task failure.

### Output format
- no-thinking, action only. 
- **At most one** action that interact with the environment inside `<action></action>` in the end of response. If you choose to update query, you can take no action on current step.
- Any number of memory actions with `< >`.

Example of responses:
<update_query>The action history in the art studio.</update_query><action>open art studio door</action>

<add_memory>The orange was placed in my inventory.</add_memory><action>go to the garden</action>

<modify_memory>Memory 1: The orange was no longer in my inventory, I ate it.</modify_memory><action>look around</action>

Your recent action history(Be careful not to repeat similar actions):
{history}

"""

class SCIMemAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + 4096
        logger.info(f'\n[RECURRENT] max_input_length: = {self.max_input_length}\n')
        logger.info(f"chunk_size: {self.config.chunk_size}")
        self.NO_MEMORY_TOKENS = "No previous memory"
    
    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.bsz = len(gen_batch.non_tensor_batch["env_name"])
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        self.is_final = torch.zeros(self.bsz, dtype=torch.bool)
        self.faiss_index = faiss.IndexFlatL2(1024)
        self.vectorstore = FAISS(self.call_remote_embedding, self.faiss_index, docstore= InMemoryDocstore(),
                index_to_docstore_id={})
        self.agent_step = [0]*self.bsz
        self.temp_memory = [[]]*self.bsz
        self.query = ["No query"]*self.bsz
        self.query_times = [0]*self.bsz
        self.history = [""]*self.bsz
        self.observations = [""]*self.bsz
        self.reward = [0.0]*self.bsz
        total_env_infos = []
        for idx in range(self.bsz):
            env_name = self.gen_batch.non_tensor_batch["env_name"][idx]
            var = self.gen_batch.non_tensor_batch["var"][idx]
            env_info = {
                "env_name": env_name,
                "var": var
            }
            total_env_infos.append(env_info)
        self.manager = SCIWorldMultiEnvManager(total_env_infos)
        initial_feedbacks = self.manager.init_envs()
        self.prompt = []
        for init_res in initial_feedbacks:
            self.prompt.append(init_res["prompt"])
        assert len(self.prompt) == self.bsz


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
            message = [{"role": "user", "content": TEMPLATE.format(prompt = prompt, query = self.query[idx], memory=memory_text, history=self.history[idx][-self.config.max_sliding_window_length:], max_chunk=self.config.max_chunks, agent_step=self.agent_step[idx])}]
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
            task_buffer = {"running_ids": [], "responses": []}
            for idx, real_idx in enumerate(true_indices):
                text = self.tokenizer.decode(texts[idx], skip_special_tokens=True)
                self.history[real_idx] += "\n--------\n" + text
                self.parse_memory_action(real_idx, text, action_rate_metric)
                self.agent_step[real_idx] += 1
                action, is_valid = self.parse_action(text)
                if is_valid:
                    if action:
                        task_buffer["running_ids"].append(real_idx)
                        task_buffer["responses"].append(action)
                else:
                    self.history[real_idx] += '\nThe action format is not valid.'
            feedbacks = self.manager.execute_actions(**task_buffer)
            for feedback in feedbacks:
                running_id = feedback["running_id"]
                exe_res = feedback["exe_res"]
                self.observations[running_id] = exe_res["observation"]
                self.history[running_id] += '\n'+ exe_res["observation"]
                if exe_res["reward"] != -1:
                    self.reward[running_id] = exe_res["reward"]
                self.is_final[running_id] = exe_res["isCompleted"]
                # 超出最大步数直接结束，防止无限循环
            for real_idx in true_indices:
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
        del self.meta_info
        del self.messages
        self.dump_vectorstore("/home/test/test03/huoyupeng/MemAgent/MemAgent-main/dataset.txt")
        del self.vectorstore
        del self.faiss_index
        del self.query
        self.manager.shutdown()
        del self.manager
        del self.prompt
        del self.history
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index, self.reward
        

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
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")
            
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
    
    def parse_action(self, text):
        pattern = re.compile(r'<action[^>]*>(.*?)</action>', flags=re.S | re.I)
        tags = pattern.findall(text)
        query_pattern = re.compile(r'<update_query[^>]*>(.*?)</update_query>', flags=re.S | re.I)
        query_tags = query_pattern.findall(text)
        if len(tags) == 0 or len(tags) > 1:
            if len(query_tags)== 0:
                return None, False
            else:
                return None, True
        else:
            action = tags[0]
            return action, True

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
        


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=SCIMemAgent)
