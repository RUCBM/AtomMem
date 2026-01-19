import re
import json

REWARD_PROMPT = '''
You are an impartial evaluator.  
I will give you two answers to the same question:

[RESPONSE]
{response}

[GROUND_TRUTH]
{ground_truth}

Your task:
Compare the two answers: if they convey the same meaning (allowing different wording, order, or extra details), output ```json{{"pass": true}}```, otherwise output ```json{{"pass": false}}```. As long as the response contains important information that is in the ground truth (e.g., numbers), it should be judged as true; redundant information in the response should be ignored.
'''

def compute_score(solution_str, ground_truth: list|str) -> float: 
    def compute_score_single(solution_str, ground_truth) -> float:
        ground_truth = ground_truth.lower()

        retval = 0.
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
                if is_equiv(answer, ground_truth):
                    retval = 1.
        except Exception as e:
            print(e)
        return retval
    solution_str = solution_str[-300:].lower()
    if isinstance(ground_truth, list):
        return max(compute_score_single(solution_str, gt) for gt in ground_truth)
    elif isinstance(ground_truth, str):
        return compute_score_single(solution_str, ground_truth)

def compute_score_LLM(client, model_name, solution_str, ground_truth: list|str) -> float:
    def compute_score_single(client, solution_str, ground_truth) -> float:
        ground_truth = ground_truth.lower()

        retval = 0.
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
                if is_equiv_LLM(client, answer, ground_truth, model_name):
                    retval = 1.
        except Exception as e:
            print(e)
        return retval
    solution_str = solution_str[-300:].lower()
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            score = compute_score_single(client, solution_str, gt)
            if score > 0:
                ground_truth.remove(gt)
                return score
        return 0.
    elif isinstance(ground_truth, str):
        return compute_score_single(client, solution_str, ground_truth)


def is_equiv_LLM(client, answer, ground_truth, eval_model_name):
    prompt = [{"role": "user",
                   "content": REWARD_PROMPT.format(response=answer, ground_truth=ground_truth)}]
    res = client.chat.completions.create(
        model=eval_model_name,
        messages=prompt,
        temperature=0.6,
        max_tokens=2048,
        timeout=1800,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    response = res.choices[0].message.content
    try:
        pattern = r"```json(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        json_str = match.group(1).strip()   # 提取字符串
        json_res = json.loads(json_str)
        if json_res["pass"]:
            return 1
        else:
            return 0
    except:
        return 0


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    # remove spaces
    string = string.replace(" ", "")

    return string
