from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

from nano_verl.config import OptimizerConfig

class Engine:
    def __init__(
        self,
        model_name: str,
        optimizer_config: OptimizerConfig,
    ):
        self.model_name = model_name
        self.optimizer_config = optimizer_config

        unwrapped_model = AutoModelForCausalLM(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa" # hope that this torch+transformers setup has FA4 already
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        self.model = FSDP(
            unwrapped_model,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

        self.optimizer = self.optimizer_config.get_optimizer(self.model)
        lr_scheduler_func = self.optimizer_config.get_lr_scheduler()
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_scheduler_func)

    def forward_log_probs(self, full_ids: torch.Tensor, full_attention_mask: torch.Tensor, response_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
        ).logits

        logits = logits[:, :-1, :]
        labels = full_ids[:, 1:]
        mask = response_mask[:, 1:]

        log_probs = F.log_softmax(logits, labels, dim=-1)
        per_token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        return per_token_log_probs, mask

    def train_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def get_weights(self):
        return [
            (name, param) for name, param in self.model.named_parameters()
        ]

