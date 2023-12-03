# script containing relevant metrics
import torch
from typing import Union
@torch.no_grad()
def gan_representation_error(
    G_z : torch.Tensor,
    x   : torch.Tensor,
    p   : Union[int, float] = 2,
) -> float:
    # compute image residual
    res = G_z - x
    return torch.norm(res.flatten(), p = p)
