from typing import List, Tuple

import torch
from torch import Tensor

from oml.interfaces.miners import ITripletsMinerInBatch
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner


class PairsMiner:
    # todo: add this class to registry, probably add BCEWithPairsMiner as well

    _miner: ITripletsMinerInBatch

    def __init__(self, hard_mining: bool = True):
        super().__init__()
        self._miner = HardTripletsMiner() if hard_mining else AllTripletsMiner()

    def sample(self, features: Tensor, labels: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        ii_a, ii_p, ii_n = self._miner._sample(features, labels=labels)

        # todo: refactor the current approach when we create pairs disassembling triplets
        ii_a_1, ii_p = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_p))))))
        ii_a_2, ii_n = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_n))))))

        is_negative = torch.ones(len(ii_a_1) + len(ii_a_2)).bool()
        is_negative[: len(ii_a_1)] = 0

        return torch.tensor([*ii_a_1, *ii_a_2]).long(), torch.tensor([*ii_p, *ii_n]).long(), is_negative


__all__ = ["PairsMiner"]
