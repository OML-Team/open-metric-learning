from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import ITripletsMinerInBatch, TTripletsIds
from oml.utils.misc_torch import pairwise_dist


class NHardTripletsMiner(ITripletsMinerInBatch):
    """
    This miner selects hard triplets based on distances between features:

    - hard `positive` samples have large distance to the anchor sample

    - hard `negative` samples have small distance to the anchor sample

    Toward the end of the training, annotation errors can affect final metric. If you are not sure about the quality of
    your dataset, you can use range instead of integer value for parameters and exclude combinations with the largest
    distances. For example instead picking 5 positive examples, you can use examples from the 2nd hardest
    to the 5th one.

    """

    def __init__(
        self,
        n_positive: Union[Tuple[int, int], List[int], int] = 1,
        n_negative: Union[Tuple[int, int], List[int], int] = 1,
    ):
        """
        Args:
            n_positive: keep ``n_positive`` positive examples with large distances. If the value is a range, minimal
                value has to be less than the available amount of labels in batches
            n_negative: keep ``n_negative`` negative examples with small distances

        Note:
            If both parameters are 1, the miner is equivalent to ``HardTripletsMiner``.
            If both parameters are large enough, the miner can be equivalent to ``AllTripletsMiner``

        """

        self.positive_slice = slice(*self._parse_input_arg(n_positive))
        self.negative_slice = slice(*self._parse_input_arg(n_negative))

    @staticmethod
    def _parse_input_arg(n: Union[Tuple[int, int], List[int], int]) -> List[int]:
        if isinstance(n, int):
            n = [0, n]
        elif isinstance(n, (list, tuple)):
            n = list(n)
            n[0] -= 1
        else:
            raise TypeError("Unsupported type of argument. Must be int, tuple or list")

        assert n[1] > n[0]
        assert n[0] >= 0

        return n

    def _sample(
        self,
        features: Tensor,
        labels: List[int],
        *_: Any,
        ignore_anchor_mask: Optional[Union[List[int], Tensor, np.ndarray]] = None,
    ) -> TTripletsIds:
        """
        This method samples the hard triplets inside the batch based on distance between features.

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``
            ignore_anchor_mask: Parameter allows you to specify the ids of features that cannot be used as anchors. Can
            be useful with memory banks to create triplets in which at least one vector will have gradients.

        Returns:
            The batch of the triplets in the order below:
            ``(anchor, positive, negative)``
        """
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

        ids_highest_distance = torch.argsort(distmat_reduced, descending=True, dim=1)
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
            positives = hardest_positive[idx_anch_pos == idx_anch][self.positive_slice]
            negatives = hardest_negative[idx_anch_neg == idx_anch][self.negative_slice]

            i_pos, i_neg = list(zip(*torch.cartesian_prod(positives, negatives).tolist()))
            ids_a.extend([idx_anch.item()] * len(i_pos))
            ids_p.extend(i_pos)
            ids_n.extend(i_neg)

        return ids_a, ids_p, ids_n
