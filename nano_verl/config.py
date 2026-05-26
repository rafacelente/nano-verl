from pydantic import BaseModel
import torch.nn as nn

from nano_verl.utils.optimizers import get_optimizer, get_lr_scheduler_func

class OptimizerConfig(BaseModel):
    learning_rate: float
    optimizer: str = "adamw"
    lr_scheduler: str = "constant"

    def get_optimizer(self, model: nn.Module, **optimizer_kwargs):
        return get_optimizer(self.optimizer, model=model, lr=self.learning_rate, **optimizer_kwargs)
    
    def get_lr_scheduler(self, **lr_scheduler_kwargs):
        return get_lr_scheduler_func(self.lr_scheduler, **lr_scheduler_kwargs)

    