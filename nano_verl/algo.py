from typing import Optional, Union
from collections import defaultdict
import torch
import numpy as np

from nano_verl.dataproto import Batch

def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: Optional[Union[int, tuple[int, ...]]] = None) -> torch.Tensor:
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[Union[int, tuple[int, ...]]] = None) -> torch.Tensor:
    sum = masked_sum(values, mask, axis)
    return sum / (mask.sum(axis=axis) + 1e-8)

def compute_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    scores = token_level_rewards.sum(dim=-1) # [bs,]

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        batch_size = scores.shape[0]
        for i in range(batch_size):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        for i in range(batch_size):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        
        scores = scores.unsqueeze(-1) * response_mask # broadcast to all tokens
    
    return scores


def compute_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio: float = 0.2,
):
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20, max=20.0)
    ratio = torch.exp(negative_approx_kl) # \pi_\theta / \pi_\theta_k
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    loss_per_token = torch.max(unclipped_loss, clipped_loss)
    loss = masked_mean(loss_per_token, response_mask)
    return loss

def compute_kl_loss(
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
):
    kl = new_log_probs - ref_log_probs
    return masked_mean(kl, response_mask)


def grpo_step(
    model,
    ref_model,
    batch: Batch,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.001,
):
    token_level_rewards = batch.tensors["rewards"].unsqueeze(-1) * batch.tensors["response_mask"]
    
    advantages = compute_grpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=batch.tensors["response_mask"],
        index=batch.metadata["uids"],
    )

    new_log_probs_full = model.forward_log_probs(
        full_ids=batch.tensors["full_ids"],
        full_attention_mask=batch.tensors["full_attention_mask"]
    )

    with torch.no_grad():
        ref_log_probs_full = ref_model.forward_log_probs(
            full_ids=batch.tensors["full_ids"],
            full_attention_mask=batch.tensors["full_attention_mask"]
        )
    start = batch.tensors["attention_mask"].shape[1] - 1
    new_log_probs = new_log_probs_full[:, start:] # (bs, resp_length)
    ref_log_probs = ref_log_probs_full[:, start:] # (bs, resp_length)
    
    old_log_probs = batch.tensors["old_log_probs"] # (bs, resp_length)
    mask = batch.tensors["response_mask"]

    policy_loss = compute_policy_loss(
        old_log_probs, new_log_probs, advantages, mask, clip_ratio,
    )

    kl_loss = compute_kl_loss(new_log_probs, ref_log_probs, mask)

    return policy_loss + kl_loss * kl_coef 

