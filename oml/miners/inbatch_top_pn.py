from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import ITripletsMinerInBatch, TTripletsIds
from oml.utils.misc_torch import pairwise_dist


class TopPNTripletsMiner(ITripletsMinerInBatch):
    """
    This miner selects hard triplets based on distances between features:
    hard positive sample has large distance to the anchor sample,
    hard negative sample has small distance to the anchor sample.
    """

    def __init__(
        self,
        top_positive: Union[Tuple[int, int], List[int], int] = 1,
        top_negative: Union[Tuple[int, int], List[int], int] = 1,
    ):
        """
        Args:
            top_positive: keep positive examples with largest distance
            top_negative: keep negative examples with smallest distance

        Notes: Toward the end of the training, annotation errors can affect, if you are not sure about the quality
        of your dataset, you can use range instead of integer value for paramters.
        """

        self.top_positive_slice = slice(*self._parse_input_arg(top_positive))
        self.top_negative_slice = slice(*self._parse_input_arg(top_negative))

    @staticmethod
    def _parse_input_arg(top: Union[Tuple[int, int], List[int], int]) -> List[int]:
        available_types = (list, tuple, int)
        assert isinstance(top, available_types), f"Unsupported type of argument, must be only {available_types}"

        if isinstance(top, int):
            top = [0, top]
        else:
            top = list(top)
            top[0] -= 1

        assert top[1] > top[0]
        assert top[0] >= 0

        return top

    def _sample(
        self,
        features: Tensor,
        labels: List[int],
        *_: Any,
        ignore_anchor_mask: Optional[Union[List[int], Tensor, np.ndarray]] = None,
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
        ignore_anchor_mask: Optional[Union[List[int], Tensor, np.ndarray]] = None,
    ) -> TTripletsIds:
        if isinstance(labels, (list, tuple)):
            labels = torch.tensor(labels, device=distmat.device).long()

        if ignore_anchor_mask is None:
            ignore_anchor_mask = torch.zeros(len(distmat), dtype=torch.bool, device=distmat.device)

        all_ids_reduced = torch.arange(len(labels), device=distmat.device)[torch.logical_not(ignore_anchor_mask)]

        distmat_reduced = distmat[torch.logical_not(ignore_anchor_mask)]
        ii_arange = (
            torch.arange(len(distmat_reduced), device=distmat.device).unsqueeze(-1).expand(distmat_reduced.shape)
        )

        _, ids_highest_distance = torch.topk(distmat_reduced, k=distmat_reduced.shape[-1], largest=True)
        mask_same_label = labels[:, None] == labels[None, :]  # type: ignore
        mask_same_label.fill_diagonal_(False)
        mask_same_label = mask_same_label[torch.logical_not(ignore_anchor_mask)]
        mask_same_label_sorted_by_dist = mask_same_label[ii_arange, ids_highest_distance]
        idx_anch_pos_reduced, idx_pos_sorted_by_dist = torch.nonzero(mask_same_label_sorted_by_dist, as_tuple=True)
        hardest_positive = ids_highest_distance[idx_anch_pos_reduced, idx_pos_sorted_by_dist]
        idx_anch_pos = all_ids_reduced[idx_anch_pos_reduced]

        ids_smallest_distance = torch.flip(ids_highest_distance, dims=(1,))
        mask_diff_label = labels[:, None] != labels[None, :]  # type: ignore
        mask_diff_label = mask_diff_label[torch.logical_not(ignore_anchor_mask)]
        diff_label_sorted_by_dist = mask_diff_label[ii_arange, ids_smallest_distance]
        idx_anch_neg_reduced, idx_neg_sorted_by_dist = torch.nonzero(diff_label_sorted_by_dist, as_tuple=True)
        hardest_negative = ids_smallest_distance[idx_anch_neg_reduced, idx_neg_sorted_by_dist]
        idx_anch_neg = all_ids_reduced[idx_anch_neg_reduced]

        ids_a = []
        ids_p = []
        ids_n = []

        for idx_anch in torch.arange(len(labels))[torch.logical_not(ignore_anchor_mask)]:
            positives = hardest_positive[idx_anch_pos == idx_anch][self.top_positive_slice]
            negatives = hardest_negative[idx_anch_neg == idx_anch][self.top_negative_slice]

            i_pos, i_neg = list(zip(*torch.cartesian_prod(positives, negatives).tolist()))
            ids_a.extend([idx_anch.item()] * len(i_pos))
            ids_p.extend(i_pos)
            ids_n.extend(i_neg)

        return ids_a, ids_p, ids_n
