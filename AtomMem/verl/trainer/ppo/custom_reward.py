from verl.utils.reward_score import hotpotqa
def compute_score(data_source, ground_truth, conversation, final_conversation, extra_info=None, **kwargs):
    solution_str = final_conversation[1]
    raw_reward = hotpotqa.compute_score(solution_str, ground_truth)
    frequence_penalty_rate = 0.1
    total = 0
    update = 0
    for conv in conversation:
        total += 1
        if "update memory" in conv[1]:
            update += 1
    frequence = update/total
    reward = raw_reward - frequence_penalty_rate * frequence
    return reward
    