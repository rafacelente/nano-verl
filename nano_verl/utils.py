from typing import Optional, Union
import torch

def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: Optional[Union[int, tuple[int, ...]]] = None) -> torch.Tensor:
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[Union[int, tuple[int, ...]]] = None) -> torch.Tensor:
    sum = masked_sum(values, mask, axis)
    return sum / (mask.sum(axis=axis) + 1e-8)