from itertools import combinations, product
from operator import itemgetter
from random import sample
from sys import maxsize
from typing import List

import numpy as np
import torch

from oml.interfaces.miners import ITripletsMinerInBatch, TLabels, TTripletsIds
from oml.utils.misc import find_value_ids


class AllTripletsMiner(ITripletsMinerInBatch):
    """
    This miner selects all the possible triplets for the given batch.

    """

    def __init__(self, max_output_triplets: int = maxsize, device: str = "cpu"):
        """
        Args:
            max_output_triplets: Number of all of the possible triplets
              in the batch can be very large, so we can limit them vis this parameter.

            device: the device where to perform computations.

        """
        self._max_out_triplets = max_output_triplets
        self._device = device

    def _sample(self, *_: torch.Tensor, labels: List[int]) -> TTripletsIds:  # type: ignore
        """
        Args:
            labels: Labels of the samples in the batch
            *_: Note, that we ignore features argument

        Returns:
            Indices of the triplets

        """
        return get_available_triplets(labels, max_out_triplets=self._max_out_triplets, device=self._device)


def get_available_triplets(labels: TLabels, max_out_triplets: int = maxsize, device: str = "cpu") -> TTripletsIds:
    """
    Generate a random subset of size ``max_out_triplets`` from the set of
    all possible triplets (anchor, positive, negative).

    Args:
        labels: Labels of the samples.
        max_out_triplets: maximal number of triplets to generate
        device: the desired device of returned tensor

    Returns:
        lists of indexes of the anchors, positives and negatives.

    """
    labels = torch.tensor(labels, device=device, requires_grad=False)

    # First we need to get initial ingridients of triplets:
    # the arrays of all (anchor, positive) and (anchor, negavite) pairs
    mask_same_label: torch.Tensor = labels[:, None] == labels[None, :]
    idx_anch_neg = torch.nonzero(torch.logical_not(mask_same_label))
    mask_same_label.fill_diagonal_(False)
    idx_anch_pos = torch.nonzero(mask_same_label)
    del mask_same_label

    # Next we need to group (anchor, negative) pairs
    # with respect to the anchor. As a result we get an array called neg_groups,
    # such that neg_groups[i] is all the negatives, that come along with
    # the anchor number i.
    # For simplicity, lets assume that
    # idx_anch_pos = [[0, 1],
    #                 [0, 2],
    #                 [1, 3],
    #                 [1, 4]]
    # and idx_anch_neg = [[0, 5],
    #                     [0, 6],
    #                     [1, 7]]
    #                     [1, 8]]
    # We want to get the following correspondence:
    #     0 -> [5, 6]
    #     1 -> [7, 8]
    # We know that idx_neg is sorted lexicographically. So, we can fint the number
    # of all unique anchors in idx_anch_neg, and use torch.split to get the desired
    # groups.
    _, neg_counts = torch.unique(idx_anch_neg[:, 0], return_counts=True, sorted=True)
    neg_groups = torch.split(idx_anch_neg[:, 1], neg_counts.tolist())

    # Now for a pair (anchor, positive)we have
    # neg_group[i], where i is the number of the anchor, with all
    # the negatives that come along with that anchor.
    # In order to get all possible triplets that contains (anchor, positive)
    # we just need to repeat it as much as we have negative elements
    # in the neg_groups[i].
    # For example above, we want to generate the following triplets:
    #
    # we need to repeat [0, 1] as much as there are | [0, 1, 5]  |
    # negatives that come along with anchor 0       | [0, 1, 6]  | and we repeat [5, 6] as much
    #                                                 [0, 2, 5]  | as there are anchor 0 in idx_anch_pos
    #                                                 [0, 2, 6]  |
    #                                                 [1, 3, 7]
    #                                                 [1, 3, 8]
    #                                                 [1, 4, 7]
    #                                                 [1, 4, 8]

    # Repeat the negatives as much as there are correspondent anchors.
    idx_neg = torch.cat(tuple(itemgetter(*idx_anch_pos[:, 0].cpu().tolist())(neg_groups)))

    # Repeat the positive pairs as much as there are correspondent netatives and get idx_anch, idx_pos.
    n_negs_in_group = [len(g) for g in neg_groups]
    n_repeats = torch.tensor(tuple(itemgetter(*idx_anch_pos[:, 0].cpu().numpy())(n_negs_in_group))).to(idx_anch_pos)
    idx_anch_pos = idx_anch_pos.repeat_interleave(n_repeats, dim=0)
    idx_anch, idx_pos = idx_anch_pos.T

    # Choose randomly the required number of triplets.
    random_idx = np.random.choice(len(idx_anch), min(len(idx_anch), max_out_triplets), replace=False)
    random_idx = torch.tensor(random_idx).to(device)
    idx_anch = idx_anch[random_idx]
    idx_pos = idx_pos[random_idx]
    idx_neg = idx_neg[random_idx]

    idx_anch = idx_anch.tolist()
    idx_pos = idx_pos.tolist()
    idx_neg = idx_neg.tolist()

    return idx_anch, idx_pos, idx_neg


def get_available_triplets_naive(labels: List[int], max_out_triplets: int = maxsize) -> TTripletsIds:
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
