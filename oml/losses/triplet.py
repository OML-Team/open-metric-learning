from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from oml.functional.losses import get_reduced
from oml.interfaces.criterions import ITripletLossWithMiner
from oml.interfaces.miners import ITripletsMiner, labels2list
from oml.miners.cross_batch import TripletMinerWithMemory
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.utils.misc_torch import elementwise_dist

TLogs = Dict[str, float]


class TripletLoss(Module):
    """
    Class, which combines classical `TripletMarginLoss` and `SoftTripletLoss`.
    The idea of `SoftTripletLoss` is the following:
    instead of using the classical formula
    ``loss = relu(margin + positive_distance - negative_distance)``
    we use
    ``loss = log1p(exp(positive_distance - negative_distance))``.
    It may help to solve the often problem when `TripletMarginLoss` converges to it's
    margin value (also known as `dimension collapse`).

    """

    criterion_name = "triplet"  # for better logging

    def __init__(self, margin: Optional[float], reduction: str = "mean", need_logs: bool = False):
        """

        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            reduction: ``mean``, ``sum`` or ``none``
            need_logs: Set ``True`` if you want to store logs

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(TripletLoss, self).__init__()

        self.margin = margin
        self.reduction = reduction
        self.need_logs = need_logs
        self.last_logs: Dict[str, float] = {}

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """

        Args:
            anchor: Anchor features with the shape of ``(batch_size, feat)``
            positive: Positive features with the shape of ``(batch_size, feat)``
            negative: Negative features with the shape of ``(batch_size, feat)``

        Returns:
            Loss value

        """
        assert anchor.shape == positive.shape == negative.shape

        positive_dist = elementwise_dist(x1=anchor, x2=positive, p=2)
        negative_dist = elementwise_dist(x1=anchor, x2=negative, p=2)

        if self.margin is None:
            # here is the soft version of TripletLoss without margin
            loss = torch.log1p(torch.exp(positive_dist - negative_dist))
        else:
            loss = torch.relu(self.margin + positive_dist - negative_dist)

        if self.need_logs:
            self.last_logs = {
                "active_tri": float((loss.clone().detach() > 0).float().mean()),
                "pos_dist": float(positive_dist.clone().detach().mean().item()),
                "neg_dist": float(negative_dist.clone().detach().mean().item()),
            }

        loss = get_reduced(loss, reduction=self.reduction)

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
    """
    The same as `TripletLoss`, but works with anchor, positive and negative features stacked together.

    """

    criterion_name = "triplet"  # for better logging

    def __init__(self, margin: Optional[float], reduction: str = "mean", need_logs: bool = False):
        """

        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            reduction: ``mean``, ``sum`` or ``none``
            need_logs: Set ``True`` if you want to store logs

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(TripletLossPlain, self).__init__()
        self.criterion = TripletLoss(margin=margin, reduction=reduction, need_logs=need_logs)
        self.last_logs = self.criterion.last_logs

    def forward(self, features: torch.Tensor) -> Tensor:
        """

        Args:
            features: Features with the shape of ``[batch_size, feat]`` with the following structure:
                      `0,1,2` are indices of the 1st triplet,
                      `3,4,5` are indices of the 2nd triplet,
                      and so on.
                      Thus, the features contains ``(N / 3)`` triplets

        Returns:
            Loss value

        """
        n = len(features)
        assert n % 3 == 0

        anchor_ii, positive_ii, negative_ii = get_tri_ids_in_plain(n)

        return self.criterion(features[anchor_ii], features[positive_ii], features[negative_ii])


class TripletLossWithMiner(ITripletLossWithMiner):
    """
    This class combines `Miner` and `TripletLoss`.

    """

    criterion_name = "triplet"  # for better logging

    def __init__(
        self,
        margin: Optional[float],
        miner: ITripletsMiner = AllTripletsMiner(),
        reduction: str = "mean",
        need_logs: bool = False,
    ):
        """

        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            miner: A miner that implements the logic of picking triplets to pass them to the triplet loss.
            reduction: ``mean``, ``sum`` or ``none``
            need_logs: Set ``True`` if you want to store logs

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super().__init__()
        self.tri_loss = TripletLoss(margin=margin, reduction="none", need_logs=need_logs)
        self.miner = miner
        self.reduction = reduction
        self.need_logs = need_logs

        self.last_logs: Dict[str, float] = {}

    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tensor:
        """
        Args:
            features: Features with the shape ``[batch_size, feat]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Loss value

        """
        labels_list = labels2list(labels)

        # if miner can produce triplets using samples outside of the batch,
        # it has to return the corresponding indicator names <is_original_tri>
        if isinstance(self.miner, TripletMinerWithMemory):
            anchor, positive, negative, is_orig_tri = self.miner.sample(features=features, labels=labels_list)
            loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)

            if self.need_logs:

                def avg_d(x1: Tensor, x2: Tensor) -> Tensor:
                    return elementwise_dist(x1.clone().detach(), x2.clone().detach(), 2).mean()

                is_bank_tri = ~is_orig_tri
                active = (loss.clone().detach() > 0).float()
                self.last_logs.update(
                    {
                        "orig_active_tri": active[is_orig_tri].sum() / is_orig_tri.sum(),
                        "bank_active_tri": active[is_bank_tri].sum() / is_bank_tri.sum(),
                        "pos_dist_orig": avg_d(anchor[is_orig_tri], positive[is_orig_tri]),
                        "neg_dist_orig": avg_d(anchor[is_orig_tri], negative[is_orig_tri]),
                        "pos_dist_bank": avg_d(anchor[is_bank_tri], positive[is_bank_tri]),
                        "neg_dist_bank": avg_d(anchor[is_bank_tri], negative[is_bank_tri]),
                    }
                )

        else:
            anchor, positive, negative = self.miner.sample(features=features, labels=labels_list)
            loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)

        self.last_logs.update(self.tri_loss.last_logs)
        self.last_logs.update(getattr(self.miner, "last_logs", {}))

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError()

        return loss


__all__ = ["TLogs", "TripletLoss", "get_tri_ids_in_plain", "TripletLossPlain", "TripletLossWithMiner"]
