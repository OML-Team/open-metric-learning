from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from oml.interfaces.miners import ITripletsMiner, labels2list
from oml.miners.among_batches import TripletMinerWithMemory
from oml.miners.inbatch_all import AllTripletsMiner

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
            need_logs: set True if you want to return dict with intermediate calculations

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(TripletLoss, self).__init__()

        self.margin = margin
        self.reduction = reduction
        self.need_logs = need_logs

    def forward(
        self, anchor: Tensor, positive: Tensor, negative: Tensor, is_original_tri: Optional[Tensor] = None
    ) -> TLossOutput:
        """

        Args:
            anchor: Anchor features with the shape of (bs, features)
            positive: Positive features with the shape of (bs, features)
            negative: Negative features with the shape of (bs, features)
            is_original_tri: Optional indicator with the shape of (bs,) which defines if triplet comes
                from the original batch or from the memory bank.

        Returns:

        """
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

        if self.need_logs:
            if is_original_tri is not None:
                # todo: it makes no sense with margin=null
                is_bank_tri = ~is_original_tri
                orig_active = (loss.clone().detach()[is_original_tri] > 0).float().sum() / is_original_tri.sum()
                bank_active = (loss.clone().detach()[is_bank_tri] > 0).float().sum() / is_bank_tri.sum()
                logs = {"original_active_tri": float(orig_active), "bank_active_tri": float(bank_active)}
            else:
                active = (loss.clone().detach() > 0).float().mean()
                logs = {"active_tri": float(active)}
        else:
            logs = {}

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError()

        if self.need_logs:
            return loss, logs
        else:
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

        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super(TripletLossPlain, self).__init__()
        self.criterion = TripletLoss(margin=margin, reduction=reduction, need_logs=need_logs)

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

            reduction: Check args of
            >>> TripletLoss

            miner: Miner to form triplets inside the batch
        """
        assert reduction in ("mean", "sum", "none")
        assert (margin is None) or (margin > 0)

        super().__init__()
        self._miner = miner
        # todo: think about handling logs later
        self.tri_loss = TripletLoss(margin=margin, reduction=reduction, need_logs=True)

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
        if isinstance(self._miner, TripletMinerWithMemory):
            anchor, positive, negative, logs_miner, is_original_tri = self._miner.sample(
                features=features, labels=labels_list
            )
        else:
            anchor, positive, negative, logs_miner = self._miner.sample(features=features, labels=labels_list)
            is_original_tri = None

        loss, logs_loss = self.tri_loss(
            anchor=anchor, positive=positive, negative=negative, is_original_tri=is_original_tri
        )

        return loss, {**logs_miner, **logs_loss}
