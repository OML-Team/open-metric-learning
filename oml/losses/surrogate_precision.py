import torch
from torch import Tensor

from oml.functional.losses import surrogate_precision
from oml.functional.metrics import calc_gt_mask, calc_mask_to_ignore
from oml.utils.misc_torch import pairwise_dist


class SurrogatePrecision(torch.nn.Module):
    criterion_name = "surrogate_precision"

    def __init__(self, k: int, temperature1: float = 1.0, temperature2: float = 0.01):
        super(SurrogatePrecision, self).__init__()

        self.k = k
        self.temperature1 = temperature1
        self.temperature2 = temperature2

    def forward(self, emb: torch.Tensor, labels: Tensor) -> Tensor:
        assert len(emb) == len(labels)

        bs = len(emb)
        is_query, is_gallery = torch.ones(bs).bool(), torch.ones(bs).bool()
        distances = pairwise_dist(emb[is_query], emb[is_gallery])
        mask_gt = calc_gt_mask(is_query=is_query, is_gallery=is_gallery, labels=labels)
        mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)

        loss = 1 - surrogate_precision(distances=distances, mask_gt=mask_gt, mask_to_ignore=mask_to_ignore,
                                       t1=self.temperature1, t2=self.temperature2, k=self.k)

        return loss
