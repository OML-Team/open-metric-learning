from typing import List, Tuple

import numpy as np
import pytest
import torch
from scipy.spatial.distance import squareform
from torch import Tensor, tensor

from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.utils.misc import find_value_ids
from tests.test_oml.test_miners.conftest import generate_valid_labels
from tests.test_oml.test_miners.shared_checkers import check_triplets_consistency


@pytest.fixture()
def distmats_and_labels() -> List[Tuple[Tensor, List[int]]]:
    """
    Returns: list of distance matrices and valid labels

    """
    num_batches = 100

    labels_li = generate_valid_labels(num=num_batches)
    labels_list, _, _ = zip(*labels_li)

    distmats = []
    for labels in labels_list:
        n = len(labels)
        distmats.append(tensor(squareform(torch.rand(int(n * (n - 1) / 2)))))

    return list(zip(distmats, labels_list))


def test_hard_miner_from_features(features_and_labels) -> None:  # type: ignore
    """
    Args:
        features_and_labels: Features and valid labels

    """
    miner = HardTripletsMiner()

    for features, labels in features_and_labels:
        ids_a, ids_p, ids_n = miner._sample(features=features, labels=labels)

        check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)

        assert len(ids_a) == len(labels)


def test_hard_miner_from_dist(distmats_and_labels) -> None:  # type: ignore
    """
    Args:
        distmats_and_labels:
            List of distance matrices and valid labels

    """
    miner = HardTripletsMiner()

    for distmat, labels in distmats_and_labels:
        ids_a, ids_p, ids_n = miner._sample_from_distmat(distmat=distmat, labels=labels)

        check_triplets_are_hardest(
            ids_anchor=ids_a,
            ids_pos=ids_p,
            ids_neg=ids_n,
            labels=labels,
            distmat=distmat,
        )

        check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)

        assert len(labels) == len(ids_a)


def test_hard_miner_manual() -> None:
    """
    Test on manual example.

    """
    labels = [0, 0, 1, 1]

    dist_mat = torch.tensor(
        [
            [0.0, 0.3, 0.2, 0.4],
            [0.3, 0.0, 0.4, 0.8],
            [0.2, 0.4, 0.0, 0.5],
            [0.4, 0.8, 0.5, 0.0],
        ]
    )

    gt = {(0, 1, 2), (1, 0, 2), (2, 3, 0), (3, 2, 0)}

    miner = HardTripletsMiner()

    ids_a, ids_p, ids_n = miner._sample_from_distmat(distmat=dist_mat, labels=labels)
    predict = set(zip(ids_a, ids_p, ids_n))

    check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)

    assert len(labels) == len(ids_a)
    assert predict == gt


def check_triplets_are_hardest(
    ids_anchor: List[int],
    ids_pos: List[int],
    ids_neg: List[int],
    labels: List[int],
    distmat: Tensor,
) -> None:
    """
    Args:
        ids_anchor: Anchor indexes of selected triplets
        ids_pos: Positive indexes of selected triplets
        ids_neg: Negative indexes of selected triplets
        labels: Labels of the samples in the batch
        distmat: Distances between features

    """
    ids_all = set(range(len(labels)))

    for i_a, i_p, i_n in zip(ids_anchor, ids_pos, ids_neg):
        ids_label = set(find_value_ids(it=labels, value=labels[i_a]))

        ids_pos_cur = np.array(list(ids_label - {i_a}), int)
        ids_neg_cur = np.array(list(ids_all - ids_label), int)

        assert torch.isclose(distmat[i_a, ids_pos_cur].max(), distmat[i_a, i_p])

        assert torch.isclose(distmat[i_a, ids_neg_cur].min(), distmat[i_a, i_n])
