from collections import Counter
from typing import List

from scipy.special import binom


def check_triplets_consistency(
    ids_anchor: List[int], ids_pos: List[int], ids_neg: List[int], labels: List[int]
) -> None:
    """
    Args:
        ids_anchor: Anchor indexes of selected triplets
        ids_pos: Positive indexes of selected triplets
        ids_neg: Negative indexes of selected triplets
        labels: Labels of the samples in the batch

    """
    num_sampled_tri = len(ids_anchor)

    assert num_sampled_tri == len(ids_pos) == len(ids_neg)

    for (i_a, i_p, i_n) in zip(ids_anchor, ids_pos, ids_neg):
        assert len({i_a, i_p, i_n}) == 3
        assert labels[i_a] == labels[i_p]
        assert labels[i_a] != labels[i_n]

    unq_tri = set(zip(ids_anchor, ids_pos, ids_neg))

    assert num_sampled_tri == len(unq_tri)


def calc_n_triplets(labels: List[int]) -> int:
    labels_counts = Counter(labels).values()

    n_all_tri = 0
    for count in labels_counts:
        n_pos = 2 * binom(count, 2)
        n_neg = len(labels) - count
        n_all_tri += n_pos * n_neg

    return n_all_tri


def check_all_triplets_number(labels: List[int], num_selected_tri: int, max_tri: int) -> None:
    """
    Checks that the selection strategy for all triplets
    returns the correct number of triplets.

    Args:
        labels: List of labels
        num_selected_tri: Number of selected triplets
        max_tri: Limit on the number of selected triplets

    """
    n_all_tri = calc_n_triplets(labels=labels)
    assert num_selected_tri == n_all_tri or num_selected_tri == max_tri
