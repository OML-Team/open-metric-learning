from itertools import product
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import InBatchTripletsMiner, TTripletsIds
from oml.utils.misc import find_value_ids
from oml.utils.misc_torch import OnlineAvgDict, pairwise_dist


class TopPercentTripletsMiner(InBatchTripletsMiner):
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
        logs = OnlineAvgDict()

        ids_all = set(range(len(labels)))

        ids_anchor, ids_pos, ids_neg = [], [], []  # type: ignore

        eps = 10 * torch.finfo(torch.float32).eps

        if ignore_anchor_mask is None:
            ignore_anchor_mask = [False] * len(labels)

        for i_anch, (label, ignore_anchor) in enumerate(zip(labels, ignore_anchor_mask)):
            if ignore_anchor:
                continue
            ids_label = set(find_value_ids(it=labels, value=label))

            ids_pos_cur = np.array(list(ids_label - {i_anch}), int)
            ids_neg_cur = np.array(list(ids_all - ids_label), int)

            dismat_pos = distmat[i_anch, ids_pos_cur]
            dmin_pos, dmax_pos = dismat_pos.min(), dismat_pos.max()
            hardest_positive_mask = dismat_pos >= (dmin_pos + (dmax_pos - dmin_pos) * self.top_positive - eps)
            hardest_positive = torch.nonzero(hardest_positive_mask, as_tuple=True)[0]
            logs.update({"num_positive_pairs": len(hardest_positive)})

            distmat_neg = distmat[i_anch, ids_neg_cur]
            dmin_neg, dmax_neg = distmat_neg.min(), distmat_neg.max()
            hardest_negative_mask = distmat_neg <= (dmin_neg + (dmax_neg - dmin_neg) * (1 - self.top_negative) + eps)
            hardest_negative = torch.nonzero(hardest_negative_mask, as_tuple=True)[0]
            logs.update({"num_negative_pairs": len(hardest_negative)})

            ii_pos = [int(ids_pos_cur[idx]) for idx in hardest_positive]
            ii_neg = [int(ids_neg_cur[idx]) for idx in hardest_negative]

            i_pos, i_neg = list(zip(*product(ii_pos, ii_neg)))

            ids_anchor.extend([i_anch] * len(i_pos))
            ids_pos.extend(i_pos)
            ids_neg.extend(i_neg)

            self.last_logs = logs.get_dict_with_results()

        return ids_anchor, ids_pos, ids_neg
