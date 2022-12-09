import torch
from torch import Tensor, sigmoid


def get_reduced(tensor: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return tensor.mean()

    elif reduction == "sum":
        return tensor.sum()

    elif reduction == "none":
        return tensor

    else:
        raise ValueError(f"Unexpected type of reduction: {reduction}")


def surrogate_precision(
    distances: Tensor, mask_gt: Tensor, k: int, t1: float, t2: float, reduction: str = "mean"
) -> Tensor:
    distances_diff = distances - distances.unsqueeze(0).permute(2, 1, 0)
    rank = sigmoid(distances_diff / t2).sum(dim=0)
    precision = (sigmoid((k - rank) / t1) * mask_gt).sum(dim=1) / torch.clip(mask_gt.sum(dim=1), max=k)
    return get_reduced(precision, reduction=reduction)
