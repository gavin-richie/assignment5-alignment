import os

import torch
import json
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import random
import pickle
from unittest.mock import patch
from transformers import PreTrainedModel

from tests.conftest import batch_size


def masked_normalize(plog_probs, mask, normalize_constant, dim=None):
    return torch.sum(plog_probs.masked_fill(~mask, 0), dim=dim) / normalize_constant


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)          # Get probabilities from logits
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # Add small epsilon to avoid log(0)


def load_policy_into_vllm_instance(policy:PreTrainedModel, llm:LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def gradient_clipping(model, max_grad_norm:int=1.0):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
           grads.append(param.grad.view(-1))
    if not grads:
        return
    grads = torch.cat(grads)
    norm = torch.norm(grads)

    # 如果超过阈值，缩放所有梯度
    if norm>max_grad_norm:
        scale = max_grad_norm / norm
        for param in model.parameters():
            param.grad.data.mul_(scale)

def get_response_log_probs(model, input_ids, labels, return_token_entropy = False) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    indices = labels[..., None]
    selected_log_probs = torch.gather(log_probs, -1, indices)
    result = selected_log_probs.squeeze(-1)
    ret = {"log_probs": result}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        ret["token_entropy"] = token_entropy
    return ret

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.5):
    vllm_set_random_seed(seed)

    world_size_path = patch("torch.distributed.get_world_size", return_value=1)
    profiling_path = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)

    with world_size_path, profiling_path:
        llm = LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return llm

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer, device) -> dict[str, torch.Tensor]:
    batch_size = len(prompt_strs)
    tokenized_prompt = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs]
    tokenized_output = [tokenizer.encode(output, add_special_tokens=False) for output in output_strs]
    prompt_and_output_len = [len(prompt) + len(output) for prompt, output in zip(tokenized_prompt, tokenized_output)]
    max_len = max(prompt_and_output_len)
    input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for i in range(batch_size):
        prompt_token, output_token, all_token = tokenized_prompt[i], tokenized_output[i], tokenized_prompt[i] + tokenized_output[i]
        prompt_token_len, output_token_len, all_token_len = len(prompt_token), len(output_token), len(all_token)

        input_ids[i, :all_token_len] = torch.tensor(all_token, dtype=torch.long, device=device)
        labels[i, :all_token_len-1] = torch.tensor(all_token[1:], dtype=torch.long, device=device)
        # 3. 标记 response 部分为 True（即 prompt 之后的部分）
        response_mask[i, prompt_token_len:all_token_len] = True

    input_ids = input_ids[:,:-1]
    labels = labels[:,:-1]
    response_mask = response_mask[:,:-1]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def sft_eval(llm, num_prompts=None):
    current_dir = os.path.dirname(__file__)
    module_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(module_dir)
    math_path = os.path.join(project_root, "MATH", "sft.jsonl")
    dataset = []
    with open(math_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    if num_prompts is not None:
        dataset = random.sample(dataset, num_prompts)
    else:
        num_prompts = len(dataset)
    question_list = [item["question"] for item in dataset]
    answer_list = [data["answer"] for data in dataset]

    r1_prompt_path = os.path.join(module_dir, "prompts", "r1_zero.prompt")
    with open(r1_prompt_path, "r") as f:
        prompt_template = f.read()
    sft_prompts = [prompt_template.format(question=question) for question in question_list]

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024,
                                     stop=["</answer>"],
                                     include_stop_str_in_output=True)
    outputs = llm.evaluate(sft_prompts, sampling_params)

    generations = []
    evaluations = []
    for output, answer in zip(outputs,answer_list):
        generated_text = output.outputs[0].text
        rewards = r1_zero_reward_fn(generated_text, answer)
        generations.append(generated_text)
        evaluations.append(rewards)

    format_accuracy=sum([reward["format_reward"] for reward in evaluations])/ num_prompts
    answer_accuracy = sum([reward["answer_reward"] for reward in evaluations])/ num_prompts

    return format_accuracy, answer_accuracy