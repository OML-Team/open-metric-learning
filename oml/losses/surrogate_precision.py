import torch
from torch import Tensor

from oml.functional.losses import surrogate_precision
from oml.utils.misc_torch import pairwise_dist


class SurrogatePrecision(torch.nn.Module):
    """
    This loss is a differentiable approximation of Precision@k metric.

    The loss is described in the following paper under a bit different name:
    `Recall@k Surrogate Loss with Large Batches and Similarity Mixup`_.

    The idea is that we express the formula for Precision@k using two step functions (aka Heaviside functions).
    Then we approximate them using two sigmoid functions with temperatures.
    The smaller temperature the close sigmoid to the step function, but the gradients are sparser,
    and vice versa. In the original paper `t1 = 1.0` and `t2 = 0.01` have been used.

    .. _Recall@k Surrogate Loss with Large Batches and Similarity Mixup:
        https://arxiv.org/pdf/2108.11179v2.pdf

    """

    criterion_name = "surrogate_precision"

    def __init__(self, k: int, temperature1: float = 1.0, temperature2: float = 0.01, reduction: str = "mean"):
        """

        Args:
            k: Parameter of Precision@k.
            temperature1: Scaling factor for the 1st sigmoid, see docs above.
            temperature2: Scaling factor for the 2nd sigmoid, see docs above.
            reduction: ``mean``, ``sum`` or ``none``

        """
        super(SurrogatePrecision, self).__init__()
        assert k > 0

        """
        Note, since we consider all the batch samples as queries and galleries simultaneously,
        for each element we have its copy on the 1st position with the corresponding zero distance.
        Thus, to consider it we increase parameter k by 1.
        """

        self.k = k + 1
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.reduction = reduction

    def forward(self, features: torch.Tensor, labels: Tensor) -> Tensor:
        """

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Loss value

        """
        assert len(features) == len(labels)

        distances = pairwise_dist(x1=features, x2=features)
        mask_gt = labels[..., None] == labels[None, ...]

        loss = 1 - surrogate_precision(
            distances=distances,
            mask_gt=mask_gt,
            t1=self.temperature1,
            t2=self.temperature2,
            k=self.k,
            reduction=self.reduction,
        )

        return loss


__all__ = ["SurrogatePrecision"]
