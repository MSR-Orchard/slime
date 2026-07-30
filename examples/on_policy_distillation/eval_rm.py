from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from slime.utils.types import Sample


def get_deepscaler_rule_based_reward(response, label):
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    elif '<|return|>' in response:
        model_solution = response.split('<|return|>')[0]
    elif '<|im_end|>' in response:
        model_solution = response.split('<|im_end|>')[0]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    print("Model answer extracted:", model_answer)
    print("Ground truth label:", label, type(label))
    # assert isinstance(label, (str, float, int))
    ground_truths = [label]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    return 0


async def eval_reward_func(args, sample_or_samples, **kwargs):
    """Custom eval reward function adapted from get_deepscaler_rule_based_reward.
    
    This function is used for evaluation and follows the async_rm signature.
    It calculates rule-based reward for math problems.
    
    Args:
        args: the whole args
        sample_or_samples: Single Sample or list of Samples
        **kwargs: additional arguments
        
    Returns:
        float or list[float]: reward value(s) (0 or 1)
    """
    # Handle both single sample and batch of samples
    if isinstance(sample_or_samples, list):
        return [get_deepscaler_rule_based_reward(s.response, s.label) for s in sample_or_samples]
    else:
        return get_deepscaler_rule_based_reward(sample_or_samples.response, sample_or_samples.label)
