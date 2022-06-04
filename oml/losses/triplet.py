from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from oml.interfaces.miners import ITripletsMiner, labels2list
from oml.miners.inbatch import AllTripletsMiner


class TripletLoss(Module):
    def __init__(self, margin: Optional[float], reduction: str = "mean"):
        """

        Args:
            margin: Margin for TripletMarginLoss. Set margin=None to use SoftTripletLoss.
                    We recommend you to go give this option a chance because it may
                    solve the often problem when TripletMarginLoss converges to it's
                    margin value (dimension collapse).

            reduction: "mean" or "sum"

        """
        assert reduction in ("mean", "sum")
        assert (margin is None) or (margin > 0)

        super(TripletLoss, self).__init__()

        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        assert anchor.shape == positive.shape == negative.shape

        if len(anchor.shape) == 2:
            anchor = anchor.unsqueeze(1)
            positive = positive.unsqueeze(1)
            negative = negative.unsqueeze(1)

        positive_dist = torch.cdist(x1=anchor, x2=positive, p=2).squeeze()
        negative_dist = torch.cdist(x1=anchor, x2=negative, p=2).squeeze()

        if self.margin is None:
            # here is the soft version of TripletLoss without margin
            loss = torch.log1p(torch.exp(positive_dist - negative_dist))
        else:
            loss = torch.relu(self.margin + positive_dist - negative_dist)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError()

        return loss


def get_tri_ids_in_plain(n: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Get ids for anchor, positive and negative samples for (n / 3) triplets
    to iterate over the plain structure.

    Args:
        n: (n / 3) is the number of desired triplets.

    Returns:
        Ids of anchor, positive and negative samples
        n = 1, ret = [0], [1], [2]
        n = 3, ret = [0, 3, 6], [1, 4, 7], [2, 5, 8]

    """
    assert n % 3 == 0

    anchor_ii = list(range(0, n, 3))
    positive_ii = list(range(1, n, 3))
    negative_ii = list(range(2, n, 3))

    return anchor_ii, positive_ii, negative_ii


class TripletLossPlain(Module):
    def __init__(self, margin: Optional[float], reduction: str = "mean"):
        """
        The same as TripletLoss, but works with anchor, positive and negative
        features stacked together.

        Args:
            margin: Check args of
            >>> TripletLoss

            reduction: Check args of
            >>> TripletLoss

        """
        assert reduction in ("mean", "sum")
        assert (margin is None) or (margin > 0)

        super(TripletLossPlain, self).__init__()
        self.criterion = TripletLoss(margin=margin, reduction=reduction)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """

        Args:
            features: Features of the shape of [N, features_dim]
                      {0,1,2} are indices of the 1st triplet
                      {3,4,5} are indices of the 2nd triplet
                      and so on
                      Thus, features contain (N / 3) triplets

        Returns:
            Loss value

        """
        n = len(features)
        assert n % 3 == 0

        anchor_ii, positive_ii, negative_ii = get_tri_ids_in_plain(n)

        loss = self.criterion(features[anchor_ii], features[positive_ii], features[negative_ii])

        return loss


class TripletLossWithMiner(Module):
    """
    This class combines in-batch mining of triplets and
    computing of TripletLoss.

    """

    def __init__(self, margin: Optional[float], miner: ITripletsMiner = AllTripletsMiner(), reduction: str = "mean"):
        """
        Args:
            margin: Check args of
            >>> TripletLoss

            reduction: Check args of
            >>> TripletLoss

            miner: Miner to form triplets inside the batch
        """
        assert reduction in ("mean", "sum")
        assert (margin is None) or (margin > 0)

        super().__init__()
        self._miner = miner
        self._criterion = TripletLoss(margin=margin, reduction=reduction)

    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tensor:
        """
        Args:
            features: Features with shape [batch_size, features_dim]
            labels: Labels of samples which will be used to form triplets

        Returns:
            Loss value

        """
        labels_list = labels2list(labels)

        (
            features_anchor,
            features_positive,
            features_negative,
        ) = self._miner.sample(features=features, labels=labels_list)

        loss = self._criterion(
            anchor=features_anchor,
            positive=features_positive,
            negative=features_negative,
        )

        return loss
