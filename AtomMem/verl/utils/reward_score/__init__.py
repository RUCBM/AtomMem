# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code
import re
from openai import Client

from verl.utils.reward_score import hotpotqa
def default_compute_score(data_source, ground_truth, conversation, final_conversation, extra_info=None, **kwargs):
    if data_source in ['hotpotqa', 'gaia']:
        solution_str = final_conversation[1]
        raw_reward = hotpotqa.compute_score(solution_str, ground_truth)
    elif data_source in ['webshop', 'scienceworld']:
        raw_reward = extra_info["reward"]
    elif data_source == 'multiquery-hotpotqa':
        assert isinstance(ground_truth, list)
        flat_ground_truth = sum(ground_truth, [])
        #从前到后匹配conversation中的answer
        # 没有必要区分conversation和final_conversation，因为多轮对话中有多个答案
        conversation.append(final_conversation)
        answer_idx = 0
        raw_reward = 0
        for conv in conversation:
            ans = parse_final_answer(conv[1])
            if ans:
                raw_reward += hotpotqa.compute_score(ans[0], flat_ground_truth)
                answer_idx += 1
        raw_reward = raw_reward/len(ground_truth)
    frequence_penalty_rate = 0
    length_penalty_rate = 0.15
    total = 0
    update = 0
    total_length = 0
    if frequence_penalty_rate:
        for conv in conversation:
            total += 1
            if "update memory" in conv[1]:
                update += 1
                total_length += len(conv[1])
        frequence = update/total
    else:
        frequence = 0
    if length_penalty_rate != 0 and update != 0:
        length_penalty = total_length/(update*2048)
    else:
        length_penalty = 0
    reward = {
        "reward": raw_reward - frequence_penalty_rate * frequence - length_penalty_rate * length_penalty,
        "score":  raw_reward
    }
    if isinstance(reward, dict):
        return reward
    elif isinstance(reward, (int, float, bool)):
        return float(reward)
    else:
        return float(reward[0])
    
def default_compute_score_LLM(data_source, ground_truth, conversation, final_conversation, extra_info=None, **kwargs):
    api_key = kwargs.get("api_key", 'sk-123')
    base_url = kwargs.get("base_url", "http://localhost:8002/v1/")
    model_name = kwargs.get("model_name", 'eval')
    if data_source in ['hotpotqa','gaia']:
        solution_str = final_conversation[1]
        client = Client(api_key=api_key, base_url=base_url)
        raw_reward = hotpotqa.compute_score_LLM(client, model_name, solution_str, ground_truth)
    elif data_source in ['webshop', 'scienceworld']:
        raw_reward = extra_info["reward"]
    elif data_source == 'multiquery-hotpotqa':
        assert isinstance(ground_truth, list)
        #从前到后匹配conversation中的answer
        # 没有必要区分conversation和final_conversation，因为多轮对话中有多个答案
        flat_ground_truth = sum(ground_truth, [])
        client = Client(api_key=api_key, base_url=base_url)
        conversation.append(final_conversation)
        answer_idx = 0
        raw_reward = 0
        for conv in conversation:
            ans = parse_final_answer(conv[1])
            if ans:
                sub_reward = hotpotqa.compute_score_LLM(client, model_name, ans[0], flat_ground_truth)
                if sub_reward > 1:
                    sub_reward = 1
                raw_reward += sub_reward
                answer_idx += 1
        raw_reward = raw_reward/len(ground_truth)
        print("raw_reward:", raw_reward)
    frequence_penalty_rate = 0
    length_penalty_rate = 0
    total = 0
    update = 0
    total_length = 0
    if frequence_penalty_rate:
        for conv in conversation:
            total += 1
            if "update memory" in conv[1]:
                update += 1
                total_length += len(conv[1])
        frequence = update/total
    else:
        frequence = 0
    if length_penalty_rate != 0 and update != 0:
        length_penalty = total_length/(update*2048)
    else:
        length_penalty = 0
    reward = {
        "reward": raw_reward - frequence_penalty_rate * frequence - length_penalty_rate * length_penalty,
        "score":  raw_reward
    }
    if isinstance(reward, dict):
        return reward
    elif isinstance(reward, (int, float, bool)):
        return float(reward)
    else:
        return float(reward[0])

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'AIME', 'AMC', 'MINERVA', "MATH", 'math_dapo'] or \
            'MATH' in data_source:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['hotpotqa']:
        from . import hotpotqa
        res = hotpotqa.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])

def parse_final_answer(answer_str):
    pattern = re.compile(r'<final_answer[^>]*>(.*?)</final_answer>', flags=re.S | re.I)
    tags = pattern.findall(answer_str)
    # 确保长度为1
    if len(tags) == 1:
        return tags
    elif '\\boxed{' in answer_str:
        return [answer_str]
    else:
        return []