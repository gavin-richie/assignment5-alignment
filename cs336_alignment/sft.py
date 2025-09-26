import torch
import torch.optim as optim
import json
import random
import wandb
import argparse
from pathlib import Path
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.helpers import masked_normalize, tokenize_prompt_and_output, get_response_log_probs, log_generations, gradient_clipping
from cs336_alignment.helpers import init_vllm, load_policy_into_vllm_instance
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer


def sft_microbatch_train_step(policy_log_probs: torch.Tensor, response_mask: torch.Tensor, gradient_accumulation_steps:int,
                              normalize_constant: int)->tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    loss /= gradient_accumulation_steps
    loss.backward()

    metadata = {}
    return (loss, metadata)