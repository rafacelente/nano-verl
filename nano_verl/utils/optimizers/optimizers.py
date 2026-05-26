from typing import Union, Optional
from enum import Enum

import torch.nn as nn
import torch


class OptimizerName(str, Enum):
    ADAMW = "adamw"
    SGD = "sgd"

OPTIMIZER_MAPPING: dict[OptimizerName, type[torch.optim.Optimizer]] = {
    OptimizerName.ADAMW: torch.optim.AdamW,
    OptimizerName.SGD: torch.optim.SGD,
}

def get_optimizer(name: Union[str, OptimizerName], model: nn.Module, lr: float, **kwargs):
    optimizer_name_enum = OptimizerName(name) if isinstance(name, str) else name
    optimizer_class = OPTIMIZER_MAPPING[optimizer_name_enum]
    return optimizer_class(model.parameters(), lr=lr, **kwargs)
