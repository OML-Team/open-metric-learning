from typing import Optional

import torch
from torch import sigmoid, Tensor

from oml.functional.metrics import apply_mask_to_ignore


def surrogate_precision(distances: Tensor,
                        mask_gt: Tensor,
                        k: int,
                        t1: float,
                        t2: float,
                        mask_to_ignore: Optional[Tensor] = None
                        ):
    if mask_to_ignore:
        distances, mask_gt = apply_mask_to_ignore(distances, mask_gt, mask_to_ignore)

    distances_diff = distances - distances.unsqueeze(0).T
    rank = sigmoid(distances_diff / t2).sum(dim=0)
    precision = (sigmoid((k - rank) / t1) * mask_gt).sum(dim=1) / torch.clip(mask_gt.sum(dim=1), max=k)
    return precision.mean()
