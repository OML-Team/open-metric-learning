from typing import List

import torch
from torch import Tensor

from oml.interfaces.miners import ITripletsMinerInBatch, TTripletsIds
from oml.utils.misc_torch import pairwise_dist


class HardTripletsMiner(ITripletsMinerInBatch):
    """
    This miner selects the hardest triplets based on the distances between the features:

    - The hardest `positive` sample has the `maximal` distance to the anchor sample

    - The hardest `negative` sample has the `minimal` distance to the anchor sample

    """

    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets inside the batch.

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            The batch of the triplets in the order below:
            ``(anchor, positive, negative)``

        """
        assert features.shape[0] == len(labels)

        features = features.clone().detach()

        dist_mat = pairwise_dist(x1=features, x2=features, p=2)

        ids_anchor, ids_pos, ids_neg = self._sample_from_distmat(distmat=dist_mat, labels=labels)

        return ids_anchor, ids_pos, ids_neg

    @staticmethod
    def _sample_from_distmat(distmat: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets based on the given
        distances matrix. It chooses each sample in the batch as an
        anchor and then finds the hardest positive and negative pair.

        Args:
            distmat: Matrix with the shape of ``[batch_size, batch_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            The batch of the triplets with the size of ``batch_size``, order is the following:
                ``(anchor, positive, negative)``

        """

        labels = torch.tensor(labels)

        # Ensure labels are a torch tensor
        labels = labels.to(distmat.device)

        batch_size = labels.size(0)

        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # Shape [batch_size, batch_size]
        label_not_equal = ~label_equal

        # Get the hardest positives: argmax over the distance matrix where labels match (i.e. hardest positive)
        dist_pos = distmat.clone()
        dist_pos[label_not_equal] = -float("inf")  # Set non-positives to -inf
        hardest_pos_idx = torch.argmax(dist_pos, dim=1)

        # Get the hardest negatives: argmin over the distance matrix where labels don't match (i.e. hardest negative)
        dist_neg = distmat.clone()
        dist_neg[label_equal] = float("inf")  # Set non-negatives to +inf
        hardest_neg_idx = torch.argmin(dist_neg, dim=1)

        # Return anchor indices, positive indices, and negative indices
        ids_anchor = list(range(batch_size))
        ids_pos = hardest_pos_idx
        ids_neg = hardest_neg_idx

        return ids_anchor, ids_pos.cpu().tolist(), ids_neg.cpu().tolist()


__all__ = ["HardTripletsMiner"]
