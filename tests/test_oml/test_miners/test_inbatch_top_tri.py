from collections import defaultdict
from typing import List, Set, Tuple

import pytest
import torch

from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.miners.inbatch_top_tri import TopTripletsMiner
from oml.utils.misc_torch import pairwise_dist
from tests.test_oml.test_miners.shared_checkers import check_triplets_consistency


@pytest.mark.parametrize("top_positive", [1, 7, 17])
@pytest.mark.parametrize("top_negative", [1, 5, 13])
@pytest.mark.parametrize("duplicate_not_enough_labels", [True, False])
def test_top_tri_miner(
    features_and_labels: List[Tuple[torch.Tensor, List[int]]],
    top_positive: int,
    top_negative: int,
    duplicate_not_enough_labels: bool,
) -> None:

    miner = TopTripletsMiner(
        top_positive=top_positive, top_negative=top_negative, duplicate_not_enough_labels=duplicate_not_enough_labels
    )

    for features, labels in features_and_labels:
        labels = torch.tensor(labels)
        distmat = pairwise_dist(x1=features, x2=features, p=2)

        ids_a, ids_p, ids_n = miner._sample_from_distmat(distmat=distmat, labels=labels)

        unq_triplets = set(zip(ids_a, ids_p, ids_n))

        check_top_triplets(distmat=distmat, labels=labels, unq_triplets=unq_triplets)

        check_triplets_consistency(
            ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels, validate_duplicates=False
        )


def test_top_and_hard_are_equal(features_and_labels: List[Tuple[torch.Tensor, List[int]]]) -> None:
    miner_hard = HardTripletsMiner()
    miner_top = TopTripletsMiner(top_positive=1, top_negative=1)

    for features, labels in features_and_labels:
        triplets_from_hard = miner_hard._sample(features=features, labels=labels)
        triplets_from_top = miner_top._sample(features=features, labels=labels)

        assert triplets_from_hard == triplets_from_top


def check_top_triplets(distmat: torch.Tensor, labels: List[int], unq_triplets: Set[Tuple[int, int, int]]) -> None:
    ids_anchor2positives = defaultdict(set)
    ids_anchor2negatives = defaultdict(set)

    for anch, pos, neg in unq_triplets:
        ids_anchor2positives[anch].add(int(pos))
        ids_anchor2negatives[anch].add(int(neg))

    for idx_anch, miner_positives in ids_anchor2positives.items():
        distmat_ = distmat[idx_anch].clone()
        # we need to select largest distances across same labels and ignore self distance
        distmat_[labels != labels[idx_anch]] = -1
        distmat_[idx_anch] = -1
        _, hardest_positive = torch.topk(distmat_, k=len(miner_positives), largest=True)
        hardest_positive = set(hardest_positive.tolist())
        assert miner_positives == hardest_positive

    for idx_anch, miner_negatives in ids_anchor2negatives.items():
        distmat_ = distmat[idx_anch].clone()
        # we need to select minimal distances across another labels and ignore self distance
        distmat_[labels == labels[idx_anch]] = float("inf")
        distmat_[idx_anch] = float("inf")
        _, hardest_negative = torch.topk(distmat_, k=len(miner_negatives), largest=False)
        hardest_negative = set(hardest_negative.tolist())
        assert miner_negatives == hardest_negative
