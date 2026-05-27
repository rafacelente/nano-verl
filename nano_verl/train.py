from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nano_verl.engine import Engine
from nano_verl.dataproto import Batch
from nano_verl.generation import RolloutEngine
from nano_verl.reward import compute_score
from nano_verl.algo import grpo_step
from nano_verl.data import get_tokenized_rlvr_math_dataset, get_dataloader


def train(
    engine: Engine,
    ref_engine: Engine,
    rollout: RolloutEngine,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    config: dict[str, Any],
):
    for step, batch_dict in enumerate(dataloader):
        prompt_ids = batch_dict["input_ids"]
        prompt_masks = batch_dict["attention_mask"]
        ground_truths = batch_dict["ground_truth"]
        n = config["rollout_n"]

        num_inputs = len(prompt_ids)
        max_prompt_length = max([len(pids) for pids in prompt_ids])
        prompt_ids_padded = torch.zeros(num_inputs, max_prompt_length, dtype=torch.long)
        prompt_masks_padded = torch.zeros(num_inputs, max_prompt_length)
        uids = np.array([i for i in range(len(prompt_ids))])
        for i, (pids, pmasks) in enumerate(zip(prompt_ids, prompt_masks)):
            plen = len(pids)
            prompt_ids_padded[i, :plen] = torch.tensor(pids)
            prompt_masks_padded[i, :plen] = torch.tensor(pmasks)

        batch = Batch.from_dict(
            tensor_dict={
                "input_ids": prompt_ids_padded,
                "attention_mask": prompt_masks_padded,
            },
            metadata={
                "ground_truth": ground_truths,
                "uids": uids,
            }
        )
        batch = batch.repeat_interleave(n=n)
        bs = len(batch)

        rollout_input = [inpid.tolist() for inpid in list(batch.tensors["input_ids"])]

        rollout_output = rollout.generate(rollout_input)
        rollout_resp_lengths = [len(resp) for resp in rollout_output["response_ids"]]
        max_resp_length = max(rollout_resp_lengths)

        response_ids_padded = torch.zeros(bs, max_resp_length, dtype=torch.long)
        response_masks_padded = torch.zeros(bs, max_resp_length)
        old_log_probs_padded = torch.zeros(bs, max_resp_length)

        for i, (rids, lps) in enumerate(zip(
            rollout_output["response_ids"], rollout_output["old_log_probs"]
        )):
            rlen = len(rids)
            response_ids_padded[i, :rlen] = torch.tensor(rids)
            response_masks_padded[i, :rlen] = 1.0
            old_log_probs_padded[i, :rlen] = torch.tensor(lps)

        full_ids = torch.cat([batch.tensors["input_ids"], response_ids_padded], dim=1)
        full_attention_mask = torch.cat([batch.tensors["attention_mask"], response_masks_padded], dim=1)

        rewards = torch.zeros(bs)
        for i, (resp, gt) in enumerate(zip(rollout_output["response_ids"], batch.metadata["ground_truth"])):
            response_text = tokenizer.decode(resp, skip_special_tokens=True)
            rewards[i] = compute_score(response_text, gt)

        token_level_rewards = rewards.unsqueeze(-1) * response_masks_padded

        rollout_batch = Batch.from_dict(
            tensor_dict={
                "responses": response_ids_padded,
                "response_mask": response_masks_padded,
                "old_log_probs": old_log_probs_padded,
                "full_ids": full_ids,
                "full_attention_mask": full_attention_mask,
                "rewards": token_level_rewards
            },
            metadata={}
        )
        batch = batch.union(rollout_batch)


        

        
def mock_train(
    engine: Engine,
    ref_engine: Engine,
    rollout: RolloutEngine,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    config: dict[str, Any],
):
    for step, batch_dict in enumerate(dataloader):
        prompt_ids = batch_dict["input_ids"]
        prompt_masks = batch_dict["attention_mask"]
        ground_truths = batch_dict["ground_truth"]
        n = config["rollout_n"]

        num_inputs = len(prompt_ids)
        max_prompt_length = max([len(pids) for pids in prompt_ids])
        prompt_ids_padded = torch.zeros(num_inputs, max_prompt_length, dtype=torch.long)
        prompt_masks_padded = torch.zeros(num_inputs, max_prompt_length)
        uids = np.array([i for i in range(len(prompt_ids))])
        for i, (pids, pmasks) in enumerate(zip(prompt_ids, prompt_masks)):
            plen = len(pids)
            prompt_ids_padded[i, :plen] = torch.tensor(pids)
            prompt_masks_padded[i, :plen] = torch.tensor(pmasks)

        batch = Batch.from_dict(
            tensor_dict={
                "input_ids": prompt_ids_padded,
                "attention_mask": prompt_masks_padded,
            },
            metadata={
                "ground_truth": ground_truths,
                "uids": uids,
            }
        )
        batch = batch.repeat_interleave(n=n)
        bs = len(batch)

        rollout_input = [inpid.tolist() for inpid in list(batch.tensors["input_ids"])]

        rollout_output = rollout.generate(rollout_input)
        rollout_resp_lengths = [len(resp) for resp in rollout_output["response_ids"]]
        max_resp_length = max(rollout_resp_lengths)

        response_ids_padded = torch.zeros(bs, max_resp_length, dtype=torch.long)
        response_masks_padded = torch.zeros(bs, max_resp_length)
        old_log_probs_padded = torch.zeros(bs, max_resp_length)

        for i, (rids, lps) in enumerate(zip(
            rollout_output["response_ids"], rollout_output["old_log_probs"]
        )):
            rlen = len(rids)
            response_ids_padded[i, :rlen] = torch.tensor(rids)
            response_masks_padded[i, :rlen] = 1.0
            old_log_probs_padded[i, :rlen] = torch.tensor(lps)

        full_ids = torch.cat([batch.tensors["input_ids"], response_ids_padded], dim=1)
        full_attention_mask = torch.cat([batch.tensors["attention_mask"], response_masks_padded], dim=1)

        rewards = torch.zeros(bs)
        for i, (resp, gt) in enumerate(zip(rollout_output["response_ids"], batch.metadata["ground_truth"])):
            response_text = tokenizer.decode(resp, skip_special_tokens=True)
            rewards[i] = compute_score(response_text, gt)

        rollout_batch = Batch.from_dict(
            tensor_dict={
                "responses": response_ids_padded,
                "response_mask": response_masks_padded,
                "old_log_probs": old_log_probs_padded,
                "full_ids": full_ids,
                "full_attention_mask": full_attention_mask,
                "rewards": rewards
            },
            metadata={}
        )
        token_level_rewards = rewards.unsqueeze(-1) * response_masks_padded
        batch = batch.union(rollout_batch)

        print(batch)
        print(rewards)
        print(token_level_rewards)

        loss = grpo_step(
            engine,
            ref_engine,
            batch,
            clip_ratio=config.get("clip_ratio", 0.2),
            kl_coef=config.get("kl_coef", 0.001)
        )

        print(loss)

        

        





        

