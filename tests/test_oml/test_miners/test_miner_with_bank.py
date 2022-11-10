from itertools import chain
from random import randint, sample
from typing import List, Tuple

import pytest
import torch

from oml.miners.inbatch_nhard_tri import NHardTripletsMiner
from oml.miners.miner_with_bank import MinerWithBank


def get_features_and_labels(
    n_labels: int, n_instances: int, max_unq_labels: int, feat_dim: int, num_batches: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    assert max_unq_labels >= n_labels

    features_out = [torch.randn((n_labels * n_instances, feat_dim)) for _ in range(num_batches)]

    labels_out = []
    for _ in range(num_batches):
        labels = [[label] * n_instances for label in sample(range(max_unq_labels), k=n_labels)]
        labels = torch.tensor(list(chain(*labels))).long()
        labels_out.append(labels)

    return list(zip(features_out, labels_out))


@pytest.mark.parametrize("bank_size_in_batches", [1, 3])
@pytest.mark.parametrize("num_batches", [7, 10])
@pytest.mark.parametrize("n_labels", [2, 5])
@pytest.mark.parametrize("n_instances", [3, 6])
@pytest.mark.parametrize(
    "miner",
    [
        NHardTripletsMiner(n_positive=2, n_negative=3),
        NHardTripletsMiner(n_positive=3, n_negative=2),
    ],
)
def test_top_miner_with_bank(
    bank_size_in_batches: int,
    num_batches: int,
    n_labels: int,
    n_instances: int,
    miner: NHardTripletsMiner,
) -> None:
    """
    In this test we mimic to bank, using miners on data from current and previous batches,
    than we compare outputs of bank and mimicked bank
    """
    feat_dim = 5
    max_unq_labels = randint(n_labels, 2 * n_labels)

    top_miner_with_bank = MinerWithBank(bank_size_in_batches=bank_size_in_batches, miner=miner, need_logs=False)

    features_and_labels = get_features_and_labels(
        n_labels=n_labels,
        n_instances=n_instances,
        max_unq_labels=max_unq_labels,
        feat_dim=feat_dim,
        num_batches=num_batches,
    )

    for batch_idx in range(len(features_and_labels)):
        cur_features, cur_labels = features_and_labels[batch_idx]

        cur_and_prev_batches_features, cur_and_prev_batches_labels = list(
            zip(*features_and_labels[max(batch_idx - bank_size_in_batches, 0) : batch_idx + 1])
        )

        if batch_idx >= bank_size_in_batches:
            cur_and_prev_batches_features = torch.cat(cur_and_prev_batches_features, dim=0)
            cur_and_prev_batches_labels = torch.cat(cur_and_prev_batches_labels, dim=0)
        else:
            cur_and_prev_batches_features = cur_features
            cur_and_prev_batches_labels = cur_labels

        miner_a, miner_p, miner_n = miner.sample(
            features=cur_and_prev_batches_features, labels=cur_and_prev_batches_labels
        )
        bank_a, bank_p, bank_n = top_miner_with_bank.sample(features=cur_features, labels=cur_labels)

        # triplets can be in different order, so we have to find the same elements in loop
        len_triplet_bank = bank_a.shape[0]
        assert len_triplet_bank > 0
        # compare only triplets with anchor from current batch
        miner_triplets = torch.cat(
            [miner_a[-len_triplet_bank:], miner_p[-len_triplet_bank:], miner_n[-len_triplet_bank:]], dim=1
        )
        bank_triplets = torch.cat([bank_a, bank_p, bank_n], dim=1)

        counter_same_triplets = 0

        for bank_triplet in bank_triplets:
            counter_same_triplets += torch.any(torch.isclose(bank_triplet.unsqueeze(0), miner_triplets, atol=1e-6))

        assert len_triplet_bank == counter_same_triplets


def test_logs() -> None:
    ids_a = [0, 3, 6]
    ids_p = [1, 4, 7]
    ids_n = [2, 5, 8]
    ignore_mask = torch.Tensor([0, 1, 1, 0, 1, 1, 0, 1, 1])

    logs = MinerWithBank._prepare_logs(ids_a, ids_p, ids_n, ignore_anchor_mask=ignore_mask)

    expected_logs = {
        "positives_from_bank": 1.0,
        "positives_from_batch": 0.0,
        "negatives_from_bank": 1.0,
        "negatives_from_batch": 0.0,
    }
    assert logs == expected_logs
