from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import ITripletsMinerInBatch, TTripletsIds
from oml.utils.misc_torch import pairwise_dist


class TopPercentTripletsMiner(ITripletsMinerInBatch):
    """
    This miner selects hard triplets based on distances between features:
    hard positive sample has large distance to the anchor sample,
    hard negative sample has small distance to the anchor sample.
    """

    def __init__(self, top_positive: float = 0.9, top_negative: float = 0.9):
        """
        The number of triplets can change in each batch
        Args:
            top_positive: keep positive examples with distance which is relative close to maximal distance in batch. If
                1 then we take only example with maximal distance.
            top_negative: keep negative examples with distance which is relative close to minimal distance in batch. If
                1 then we take only example with minimal distance.
        """
        assert 0 <= top_positive <= 1
        assert 0 <= top_negative <= 1

        self.top_positive = top_positive
        self.top_negative = top_negative

        self.last_logs: Dict[str, float] = {}

    def _sample(
        self,
        features: Tensor,
        labels: List[int],
        *_: Any,
        ignore_anchor_mask: Optional[Union[List[int], Tensor, np.ndarray]] = None
    ) -> TTripletsIds:
        assert features.shape[0] == len(labels)

        dist_mat = pairwise_dist(x1=features, x2=features, p=2)

        ids_anchor, ids_pos, ids_neg = self._sample_from_distmat(
            distmat=dist_mat, labels=labels, ignore_anchor_mask=ignore_anchor_mask
        )

        return ids_anchor, ids_pos, ids_neg

    def _sample_from_distmat(
        self,
        distmat: Tensor,
        labels: List[int],
        *_: Any,
        ignore_anchor_mask: Optional[Union[List[int], Tensor, np.ndarray]] = None
    ) -> TTripletsIds:
        # logs = OnlineAvgDict()
        eps = 10 * torch.finfo(torch.float32).eps

        labels = torch.tensor(labels, device=distmat.device).long()

        if ignore_anchor_mask is None:
            ignore_anchor_mask = torch.zeros(len(distmat), dtype=torch.bool, device=distmat.device)

        all_ids_reduced = torch.arange(len(labels), device=distmat.device)[torch.logical_not(ignore_anchor_mask)]

        distmat_reduced = distmat[torch.logical_not(ignore_anchor_mask)]
        ii_arange = torch.arange(len(distmat_reduced), device=distmat.device).unsqueeze(-1).expand(distmat_reduced.shape)

        _, ids_highest_distance = torch.topk(distmat_reduced, k=distmat_reduced.shape[-1], largest=True)
        mask_same_label = labels[:, None] == labels[None, :]
        mask_same_label.fill_diagonal_(False)
        mask_same_label = mask_same_label[torch.logical_not(ignore_anchor_mask)]
        mask_same_label_sorted_by_dist = mask_same_label[ii_arange, ids_highest_distance]
        idx_anch_pos_reduced, idx_pos_sorted_by_dist = torch.nonzero(mask_same_label_sorted_by_dist, as_tuple=True)
        hardest_positive = ids_highest_distance[idx_anch_pos_reduced, idx_pos_sorted_by_dist]
        idx_anch_pos = all_ids_reduced[idx_anch_pos_reduced]

        ids_smallest_distance = torch.flip(ids_highest_distance, dims=(1,))
        mask_diff_label = labels[:, None] != labels[None, :]
        mask_diff_label = mask_diff_label[torch.logical_not(ignore_anchor_mask)]
        diff_label_sorted_by_dist = mask_diff_label[ii_arange, ids_smallest_distance]
        idx_anch_neg_reduced, idx_neg_sorted_by_dist = torch.nonzero(diff_label_sorted_by_dist, as_tuple=True)
        hardest_negative = ids_smallest_distance[idx_anch_neg_reduced, idx_neg_sorted_by_dist]
        idx_anch_neg = all_ids_reduced[idx_anch_neg_reduced]

        ids_a = []
        ids_p = []
        ids_n = []

        for idx_anch in torch.arange(len(labels))[torch.logical_not(ignore_anchor_mask)]:
            positives = hardest_positive[idx_anch_pos == idx_anch]
            negatives = hardest_negative[idx_anch_neg == idx_anch]

            distmat_pos = distmat[idx_anch, positives]
            dmin_pos, dmax_pos = distmat_pos.min(), distmat_pos.max()
            hardest_positive_mask = distmat_pos >= (dmin_pos + (dmax_pos - dmin_pos) * self.top_positive - eps)
            positives = positives[hardest_positive_mask]

            distmat_neg = distmat[idx_anch, negatives]
            dmin_neg, dmax_neg = distmat_neg.min(), distmat_neg.max()
            hardest_negative_mask = distmat_neg <= (dmin_neg + (dmax_neg - dmin_neg) * (1 - self.top_negative) + eps)
            negatives = negatives[hardest_negative_mask]

            i_pos, i_neg = list(zip(*torch.cartesian_prod(positives, negatives).tolist()))
            ids_a.extend([idx_anch] * len(i_pos))
            ids_p.extend(i_pos)
            ids_n.extend(i_neg)

        return ids_a, ids_p, ids_n
