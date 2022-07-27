from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from oml.interfaces.miners import ITripletsMiner, labels2list
from oml.miners.cross_batch import TripletMinerWithMemory
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.miners.inbatch_top_percent import TopPercentTripletsMiner
from oml.utils.misc_torch import elementwise_dist

TLogs = Dict[str, float]
TLossOutput = Union[Tensor, Tuple[Tensor, TLogs]]


class TripletLoss(Module):
    def __init__(self, margin: Optional[float], reduction: str = "mean", need_logs: bool = False):
        """

        Args:
            margin: Margin for TripletMarginLoss. Set margin=None to use SoftTripletLoss.
                    We recommend you to go give this option a chance because it may
                    solve the often problem when TripletMarginLoss converges to it's
                    margin value (dimension collapse).

            reduction: "mean", "sum" or "none"

            need_logs: Set True if you want to calculate logs.
                The result will be saved in
                >>> self.logs

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(TripletLoss, self).__init__()

        self.margin = margin
        self.reduction = reduction
        self.need_logs = need_logs
        self.last_logs: Dict[str, float] = {}

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> TLossOutput:
        """

        Args:
            anchor: Anchor features with the shape of (bs, features)
            positive: Positive features with the shape of (bs, features)
            negative: Negative features with the shape of (bs, features)

        Returns:
            loss value

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

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            return loss
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
    def __init__(self, margin: Optional[float], reduction: str = "mean", need_logs: bool = False):
        """
        The same as TripletLoss, but works with anchor, positive and negative
        features stacked together.

        Args:
            margin: Check args of
            >>> TripletLoss

            reduction: Check args of
            >>> TripletLoss

            need_logs: Set True if you want to calculate logs.
               The result will be saved in
               >>> self.last_logs

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(TripletLossPlain, self).__init__()
        self.criterion = TripletLoss(margin=margin, reduction=reduction, need_logs=need_logs)
        self.last_logs = self.criterion.last_logs

    def forward(self, features: torch.Tensor) -> TLossOutput:
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

        return self.criterion(features[anchor_ii], features[positive_ii], features[negative_ii])


class TripletLossWithMiner(Module):
    """
    This class combines in-batch mining of triplets and
    computing of TripletLoss.

    """

    def __init__(
        self,
        margin: Optional[float],
        miner: ITripletsMiner = AllTripletsMiner(),
        reduction: str = "mean",
        need_logs: bool = False,
    ):
        """
        Args:
            margin: Check args of
            >>> TripletLoss

            miner: Miner to form triplets inside the batch

            reduction: Check args of
            >>> TripletLoss

            need_logs: Set True if you want to calculate logs.
               The result will be saved in
               >>> self.last_logs
        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super().__init__()
        self.tri_loss = TripletLoss(margin=margin, reduction="none", need_logs=need_logs)
        self.miner = miner
        self.reduction = reduction
        self.need_logs = need_logs

        self.last_logs: Dict[str, float] = {}

    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tuple[Tensor, TLogs]:
        """
        Args:
            features: Features with shape [batch_size, features_dim]
            labels: Labels of samples which will be used to form triplets

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
        elif isinstance(self.miner, TopPercentTripletsMiner):
            anchor, positive, negative = self.miner.sample(features=features, labels=labels_list)
            loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)

            if self.need_logs:
                self.last_logs.update(self.miner.last_logs)
        else:
            anchor, positive, negative = self.miner.sample(features=features, labels=labels_list)
            loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)

        self.last_logs.update({k: v for k, v in self.tri_loss.last_logs.items()})

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError()

        return loss
