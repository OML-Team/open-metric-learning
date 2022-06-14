from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from oml.interfaces.miners import InBatchTripletsMiner, TTripletsIds
from oml.utils.misc import find_value_ids


class HardTripletsMiner(InBatchTripletsMiner):
    """
    This miner selects hardest triplets based on distances between features:
    the hardest positive sample has the maximal distance to the anchor sample,
    the hardest negative sample has the minimal distance to the anchor sample.

    """

    def __init__(self, norm_required: bool = False):
        """
        Args:
            norm_required: Set True if features normalisation is needed

        """
        self._norm_required = norm_required

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

        if self._norm_required:
            features = F.normalize(features.detach(), p=2, dim=1)

        dist_mat = torch.cdist(x1=features, x2=features, p=2)

        ids_anchor, ids_pos, ids_neg = self._sample_from_distmat(distmat=dist_mat, labels=labels)

        return ids_anchor, ids_pos, ids_neg

    @staticmethod
    def _sample_from_distmat(distmat: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets based on the given
        distances matrix. It chooses each sample in the batch as an
        anchor and then finds the hardest positive and negative pair.

        Args:
            distmat: Matrix of distances between the features
            labels: Labels of the samples

        Returns:
            The batch of triplets (with the size equals to the original bs)
            in the following order: (anchor, positive, negative)

        """
        ids_all = set(range(len(labels)))

        ids_anchor, ids_pos, ids_neg = [], [], []

        for i_anch, label in enumerate(labels):
            ids_label = set(find_value_ids(it=labels, value=label))

            ids_pos_cur = np.array(list(ids_label - {i_anch}), int)
            ids_neg_cur = np.array(list(ids_all - ids_label), int)

            i_pos = ids_pos_cur[distmat[i_anch, ids_pos_cur].argmax()]
            i_neg = ids_neg_cur[distmat[i_anch, ids_neg_cur].argmin()]

            ids_anchor.append(i_anch)
            ids_pos.append(i_pos)
            ids_neg.append(i_neg)

        return ids_anchor, ids_pos, ids_neg
