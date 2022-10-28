from collections import defaultdict
from itertools import chain
from random import randint, shuffle
from typing import List, Tuple, Union

import pytest
import torch

from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.miners.inbatch_top_pn import TopPNTripletsMiner
from oml.utils.misc_torch import pairwise_dist
from tests.test_oml.test_miners.shared_checkers import check_triplets_consistency

TFeaturesAndLabels = Tuple[torch.Tensor, List[int]]


@pytest.mark.parametrize("top_positive", [1, 3, (2, 4)])
@pytest.mark.parametrize("top_negative", [1, 5, (3, 6)])
def test_top_pn_miner(top_positive: Union[Tuple[int, int], int], top_negative: Union[Tuple[int, int], int]) -> None:
    miner = TopPNTripletsMiner(top_positive=top_positive, top_negative=top_negative)

    num_batches = 100
    if isinstance(top_positive, tuple):
        min_instances = top_positive[0] + 2
        max_instances = randint(min_instances, min_instances + top_positive[1])
    else:
        min_instances = 2
        max_instances = randint(min_instances, min_instances + 5)
    min_labels = 3
    max_labels = 10

    for features_and_labels in get_features_and_labels(
        num_batches=num_batches, range_labels=(min_labels, max_labels), range_instances=(min_instances, max_instances)
    ):
        check_miner(miner, features_and_labels)


@pytest.mark.parametrize(
    "miner,top_miner",
    [
        (HardTripletsMiner(), TopPNTripletsMiner(top_positive=1, top_negative=1)),
        (AllTripletsMiner(), TopPNTripletsMiner(top_positive=1000000, top_negative=1000000)),
    ],
)
def test_all_and_hard_are_specific_cases(
    miner: Union[AllTripletsMiner, HardTripletsMiner], top_miner: TopPNTripletsMiner
) -> None:
    num_batches = 100
    for features, labels in get_features_and_labels(
        num_batches=num_batches, range_labels=(2, 10), range_instances=(3, 7)
    ):
        ids_from_miner = miner._sample(features, labels=labels)
        ids_from_top_miner = top_miner._sample(features, labels=labels)

        triplets_from_miner = list(zip(*ids_from_miner))
        triplets_from_top_miner = list(zip(*ids_from_top_miner))

        assert len(triplets_from_miner) > 0
        assert len(triplets_from_top_miner) > 0

        assert set(triplets_from_miner) == set(triplets_from_top_miner)


def get_features_and_labels(
    num_batches: int, range_labels: Tuple[int, int], range_instances: Tuple[int, int], feat_dim: int = 10
) -> List[TFeaturesAndLabels]:
    labels_all = []
    features_all = []
    for _ in range(num_batches):
        labels = [[label] * randint(*range_instances) for label in range(randint(*range_labels))]
        labels = list(chain(*labels))
        shuffle(labels)
        features = torch.randn(size=(len(labels), feat_dim))

        labels_all.append(labels)
        features_all.append(features)
    return list(zip(features_all, labels_all))


def check_miner(
    top_miner: TopPNTripletsMiner,
    features_and_labels: TFeaturesAndLabels,
) -> None:
    features, labels = features_and_labels
    labels = torch.tensor(labels)
    distmat = pairwise_dist(x1=features, x2=features, p=2)

    ids_a, ids_p, ids_n = top_miner._sample_from_distmat(distmat=distmat, labels=labels)
    triplets = list(zip(ids_a, ids_p, ids_n))

    check_triplets_are_top(distmat=distmat, labels=labels, top_miner=top_miner, triplets=triplets)

    check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)


def check_triplets_are_top(
    distmat: torch.Tensor, labels: List[int], top_miner: TopPNTripletsMiner, triplets: List[Tuple[int, int, int]]
) -> None:
    ids_anchor2positives = defaultdict(set)
    ids_anchor2negatives = defaultdict(set)

    for anch, pos, neg in triplets:
        ids_anchor2positives[anch].add(int(pos))
        ids_anchor2negatives[anch].add(int(neg))

    for idx_anch, miner_positives in ids_anchor2positives.items():
        assert len(miner_positives) > 0
        distmat_ = distmat[idx_anch].clone()
        # we need to select largest distances across same labels and ignore self distance
        distmat_[labels != labels[idx_anch]] = -1
        distmat_[idx_anch] = -1
        max_available_positives = sum(labels == labels[idx_anch]) - 1  # type: ignore
        _, hardest_positive = torch.topk(distmat_, k=max_available_positives, largest=True)
        hardest_positive = set(hardest_positive[top_miner.top_positive_slice].tolist())
        assert len(miner_positives) > 0
        assert miner_positives == hardest_positive

    for idx_anch, miner_negatives in ids_anchor2negatives.items():
        assert len(miner_negatives) > 0
        distmat_ = distmat[idx_anch].clone()
        # we need to select minimal distances across another labels and ignore self distance
        distmat_[labels == labels[idx_anch]] = float("inf")
        distmat_[idx_anch] = float("inf")
        max_available_negatives = sum(labels != labels[idx_anch])  # type: ignore
        _, hardest_negative = torch.topk(distmat_, k=max_available_negatives, largest=False)
        hardest_negative = set(hardest_negative[top_miner.top_negative_slice].tolist())
        assert len(miner_negatives) > 0
        assert miner_negatives == hardest_negative
