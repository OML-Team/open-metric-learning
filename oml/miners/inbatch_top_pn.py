from itertools import product
from random import choices
from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import InBatchTripletsMiner, TTripletsIds
from oml.utils.misc import find_value_ids
from oml.utils.misc_torch import pairwise_dist


class TopPNTripletsMiner(InBatchTripletsMiner):
    """
    This miner selects hard triplets based on distances between features:
    hard positive sample has large distance to the anchor sample,
    hard negative sample has small distance to the anchor sample.
    """

    def __init__(self, top_positive: int = 1, top_negative: int = 1, duplicate_not_enough_labels: bool = False):
        """
        Args:
            top_positive: keep positive examples with topP largest distance
            top_negative: keep negative examples with topN smallest distance
            duplicate_not_enough_labels: Parameter allows automatically maintain constant number of triplets. If some
                of labels have number of instances less than top_positive or top_negative, this labels will be
                duplicated with another instances

        """
        assert top_positive >= 1
        assert isinstance(top_positive, int)
        assert top_negative >= 1
        assert isinstance(top_negative, int)

        self.top_positive = top_positive
        self.top_negative = top_negative
        self.duplicate_not_enough_labels = duplicate_not_enough_labels

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

    def _sample_from_distmat_prev(
        self,
        distmat: Tensor,
        labels: List[int],
        *_: Any,
        ignore_anchor_mask: Optional[Union[List[int], Tensor, np.ndarray]] = None
    ) -> TTripletsIds:
        ids_all = set(range(len(labels)))

        ids_anchor, ids_pos, ids_neg = [], [], []  # type: ignore
        if ignore_anchor_mask is None:
            ignore_anchor_mask = [False] * len(labels)

        for i_anch, (label, ignore_anchor) in enumerate(zip(labels, ignore_anchor_mask)):
            if ignore_anchor:
                continue
            ids_label = set(find_value_ids(it=labels, value=label))

            ids_pos_cur = np.array(list(ids_label - {i_anch}), int)
            ids_neg_cur = np.array(list(ids_all - ids_label), int)

            num_hardest_positive = min(self.top_positive, len(ids_label) - 1)
            num_hardest_negative = min(self.top_negative, len(ids_all) - len(ids_label))

            _, hardest_positive = torch.topk(distmat[i_anch, ids_pos_cur], k=num_hardest_positive, largest=True)
            _, hardest_negative = torch.topk(distmat[i_anch, ids_neg_cur], k=num_hardest_negative, largest=False)

            if self.duplicate_not_enough_labels:
                if len(hardest_positive) < self.top_positive:
                    extra_positives = torch.tensor(
                        choices(hardest_positive, k=self.top_positive - len(hardest_positive))
                    )
                    hardest_positive = torch.cat([hardest_positive, extra_positives])
                if len(hardest_negative) < self.top_negative:
                    extra_negatives = torch.tensor(
                        choices(hardest_negative, k=self.top_negative - len(hardest_negative))
                    )
                    hardest_negative = torch.cat([hardest_negative, extra_negatives])

            ii_pos = [int(ids_pos_cur[idx]) for idx in hardest_positive]
            ii_neg = [int(ids_neg_cur[idx]) for idx in hardest_negative]

            i_pos, i_neg = list(zip(*product(ii_pos, ii_neg)))

            ids_anchor.extend([i_anch] * len(i_pos))
            ids_pos.extend(i_pos)
            ids_neg.extend(i_neg)

        return ids_anchor, ids_pos, ids_neg

    def _sample_from_distmat(
        self,
        distmat: Tensor,
        labels: List[int],
        *_: Any,
        ignore_anchor_mask: Optional[Union[List[int], Tensor, np.ndarray]] = None
    ) -> TTripletsIds:

        if ignore_anchor_mask is None:
            ignore_anchor_mask = torch.zeros(len(distmat), dtype=torch.bool)

        _, ids_highest_distance = torch.topk(distmat, k=distmat.shape[-1], largest=True)

        print(ids_highest_distance.shape)
        ids_highest_distance = ids_highest_distance[[torch.logical_not(ignore_anchor_mask)]]
        print(ids_highest_distance.shape)
        ids_smallest_distance = ids_highest_distance[:, ::-1]
