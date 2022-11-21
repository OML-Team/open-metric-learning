from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, no_grad

from oml.interfaces.miners import ITripletsMiner
from oml.miners.inbatch_nhard_tri import NHardTripletsMiner
from oml.utils.misc_torch import OnlineAvgDict


class MinerWithBank(ITripletsMiner):
    """
    This is a class for cross-batch memory. This implementation uses only samples from the current batch as anchors
    and finds positive and negative pairs from the bank and the current batch using miner.

    """

    def __init__(
        self,
        bank_size_in_batches: int,
        miner: NHardTripletsMiner,
        need_logs: bool = True,
    ):
        """

        Args:
            bank_size_in_batches: Size of the bank.
            miner: Miner, for now we only support ``NHardTripletsMiner``
            need_logs: Set ``True`` if you want to track logs.

        """

        assert isinstance(bank_size_in_batches, int)
        assert bank_size_in_batches >= 1
        assert isinstance(miner, NHardTripletsMiner)
        self.miner = miner

        self.bank_size_in_batches = bank_size_in_batches
        self.bank_features: Optional[Tensor] = None
        self.bank_labels: Optional[Tensor] = None
        self.bank_size = -1  # bank size in the number of vectors
        self.is_accumulated = False  # indicates if we filled out the bank
        # We maintain a queue of feature vectors and ptr indicates the place to insert another batch.
        # After the batch has been inserted, we move the pointer.
        self.ptr = 0

        self.need_logs = need_logs
        self.last_logs: Dict[str, float] = {}

    @no_grad()
    def __allocate_if_needed(self, features: Tensor, labels: Tensor) -> None:
        if self.bank_features is None:
            assert len(features) == len(labels)

            bs = features.shape[0]
            self.feat_dim = features.shape[-1]
            self.bank_size = self.bank_size_in_batches * bs
            self.bank_labels = torch.empty(self.bank_size, dtype=torch.long, device=features.device)
            self.bank_features = torch.empty(
                (self.bank_size, self.feat_dim), dtype=features.dtype, device=features.device
            )

    @no_grad()
    def _update_bank(self, features: Tensor, labels: Tensor) -> None:
        bs = features.shape[0]
        if not self.is_accumulated:
            self.is_accumulated = self.ptr + bs >= self.bank_size

        self.bank_features[self.ptr : self.ptr + bs] = features.clone().detach()
        self.bank_labels[self.ptr : self.ptr + bs] = labels.clone()
        self.ptr = (self.ptr + bs) % self.bank_size

    def sample(self, features: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """

        Args:
            features: Features with the shape ``[batch_size, features_dim]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Batch of triplets in the following order: anchor, positive, negative

        """
        if isinstance(labels, (list, tuple)):
            labels = torch.tensor(labels, dtype=torch.long, device=features.device)

        self.__allocate_if_needed(features=features, labels=labels)

        if self.is_accumulated:
            len_batch = len(labels)
            features_miner = torch.cat([features, self.bank_features], dim=0)
            labels_miner = torch.cat([labels, self.bank_labels], dim=0)
            ignore_anchor_mask = torch.zeros(len(labels_miner), dtype=torch.bool, device=features.device)
            ignore_anchor_mask[len_batch:] = True
        else:
            features_miner = features
            labels_miner = labels
            ignore_anchor_mask = torch.zeros(len(labels_miner), dtype=torch.bool, device=features.device)

        ids_a, ids_p, ids_n = self.miner._sample(
            features=features_miner, labels=labels_miner, ignore_anchor_mask=ignore_anchor_mask
        )

        if self.need_logs:
            self.last_logs = self._prepare_logs(
                ids_a=ids_a, ids_p=ids_p, ids_n=ids_n, ignore_anchor_mask=ignore_anchor_mask
            )

        self._update_bank(features=features, labels=labels)

        return features_miner[ids_a], features_miner[ids_p], features_miner[ids_n]

    @staticmethod
    def _prepare_logs(
        ids_a: List[int], ids_p: List[int], ids_n: List[int], ignore_anchor_mask: Tensor
    ) -> Dict[str, float]:
        logs = OnlineAvgDict()

        unq_triplets = set(zip(ids_a, ids_p, ids_n))
        ids_anchor2positives = defaultdict(set)
        ids_anchor2negatives = defaultdict(set)

        for anch, pos, neg in unq_triplets:
            ids_anchor2positives[anch].add(int(pos))
            ids_anchor2negatives[anch].add(int(neg))

        for anch in ids_anchor2positives.keys():
            positives = ids_anchor2positives[anch]
            positives_from_bank = ignore_anchor_mask[list(positives)].sum().item()
            positives_from_batch = len(positives) - positives_from_bank
            logs.update({"positives_from_bank": positives_from_bank, "positives_from_batch": positives_from_batch})

            negatives = ids_anchor2negatives[anch]
            negatives_from_bank = ignore_anchor_mask[list(negatives)].sum().item()
            negatives_from_batch = len(negatives) - negatives_from_bank
            logs.update({"negatives_from_bank": negatives_from_bank, "negatives_from_batch": negatives_from_batch})

        return logs.get_dict_with_results()
