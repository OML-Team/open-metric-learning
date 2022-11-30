import torch
from torch import Tensor

from oml.functional.losses import surrogate_precision
from oml.functional.metrics import calc_gt_mask
from oml.utils.misc_torch import pairwise_dist


class SurrogatePrecision(torch.nn.Module):
    criterion_name = "surrogate_precision"

    def __init__(self, k: int, temperature1: float = 1.0, temperature2: float = 0.01, reduction: str = "mean"):
        """
        todo: docs

        https://arxiv.org/pdf/2108.11179v2.pdf
        Recall@k Surrogate Loss with Large Batches and Similarity Mixup

        Args:
            k:
            temperature1:
            temperature2:
            reduction:

        """
        super(SurrogatePrecision, self).__init__()

        self.k = k
        self.temperature1 = temperature1
        self.temperature2 = temperature2
        self.reduction = reduction

    def forward(self, emb: torch.Tensor, labels: Tensor) -> Tensor:
        assert len(emb) == len(labels)

        bs = len(emb)
        is_query = torch.ones(bs).bool().to(emb.device)
        is_gallery = torch.ones(bs).bool().to(emb.device)

        distances = pairwise_dist(x1=emb[is_query], x2=emb[is_gallery])
        mask_gt = calc_gt_mask(is_query=is_query, is_gallery=is_gallery, labels=labels)

        loss = 1 - surrogate_precision(
            distances=distances,
            mask_gt=mask_gt,
            t1=self.temperature1,
            t2=self.temperature2,
            k=self.k,
            reduction=self.reduction,
        )

        return loss
