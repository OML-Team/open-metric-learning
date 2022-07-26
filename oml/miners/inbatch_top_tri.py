from itertools import product
from random import choices
from typing import List

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import InBatchTripletsMiner, TTripletsIds
from oml.utils.misc import find_value_ids
from oml.utils.misc_torch import pairwise_dist


class TopTripletsMiner(InBatchTripletsMiner):
    """
    This miner selects hardest triplets based on distances between features:
    hard positive sample has large distance to the anchor sample,
    hard negative sample has small distance to the anchor sample.
    """

    def __init__(self, top_positive: int = 1, top_negative: int = 1, duplicate_not_enough_labels: bool = False):
        """
        Args:
            top_positive: keep positive examples with topK largest distance
            top_negative: keep negative examples with topN smallest distance
            duplicate_not_enough_labels: Parameter allows automatically maintain constant number of triplets. If some
                of labels have number of instances less than top_positive or top_negative, this labels will be
                duplicated with another

        """
        self.top_positive = top_positive
        self.top_negative = top_negative
        self.duplicate_not_enough_labels = duplicate_not_enough_labels

    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets inside the batch.

        Args:
            features: Features with the shape of [batch_size, feature_size]
            labels: Labels of the samples in the batch

        Returns:
            The batch of the triplets in the order below:
            (anchor, positive, negative)

        """
        assert features.shape[0] == len(labels)

        dist_mat = pairwise_dist(x1=features, x2=features, p=2)

        ids_anchor, ids_pos, ids_neg = self._sample_from_distmat(distmat=dist_mat, labels=labels)

        return ids_anchor, ids_pos, ids_neg

    def _sample_from_distmat(self, distmat: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets based on the given
        distances matrix. It chooses each sample in the batch as an
        anchor and then finds the hard top_positive and hard top_negatives paies.

        Args:
            distmat: Matrix of distances between the features
            labels: Labels of the samples

        Returns:
            The batch of triplets (with the size equals to the original bs)
            in the following order: (anchor, positive, negative)

        """
        ids_all = set(range(len(labels)))

        ids_anchor, ids_pos, ids_neg = [], [], []  # type: ignore

        for i_anch, label in enumerate(labels):
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
