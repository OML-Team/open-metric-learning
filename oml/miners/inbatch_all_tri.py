from itertools import combinations, product
from random import sample
from sys import maxsize
from typing import List

from torch import Tensor

from oml.interfaces.miners import ITripletsMinerInBatch, TTripletsIds
from oml.utils.misc import find_value_ids


class AllTripletsMiner(ITripletsMinerInBatch):
    """
    This miner selects all the possible triplets for the given batch.

    """

    def __init__(self, max_output_triplets: int = maxsize):
        """
        Args:
            max_output_triplets: Number of all of the possible triplets
              in the batch can be very large, so we can limit them vis this parameter.

        """
        self._max_out_triplets = max_output_triplets

    def _sample(self, *_: Tensor, labels: List[int]) -> TTripletsIds:  # type: ignore
        """
        Args:
            labels: Labels of the samples in the batch
            *_: Note, that we ignore features argument

        Returns:
            Indices of the triplets

        """
        return get_available_triplets(labels, max_out_triplets=self._max_out_triplets)


def get_available_triplets(labels: List[int], max_out_triplets: int = maxsize) -> TTripletsIds:
    """
    For each label, the function generates all possible positive and negative pairs of triplets.
    """
    num_labels = len(labels)

    triplets = []
    for label in set(labels):
        ids_pos_cur = find_value_ids(labels, label)
        ids_neg_cur = set(range(num_labels)) - set(ids_pos_cur)

        # (l0, l1, n) and (l1, l0, n) are 2 different triplets
        # and we want both of them
        pos_pairs = list(combinations(ids_pos_cur, r=2)) + list(combinations(ids_pos_cur[::-1], r=2))

        tri = [(a, p, n) for (a, p), n in product(pos_pairs, ids_neg_cur)]
        triplets.extend(tri)

    triplets = sample(triplets, min(len(triplets), max_out_triplets))
    ids_anchor, ids_pos, ids_neg = zip(*triplets)

    return list(ids_anchor), list(ids_pos), list(ids_neg)


__all__ = ["AllTripletsMiner", "get_available_triplets"]
