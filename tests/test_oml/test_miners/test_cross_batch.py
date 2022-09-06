from random import shuffle

import pytest
import torch
from torch import Tensor, tensor

from oml.miners.cross_batch import TripletMinerWithMemory
from oml.samplers.balance import BalanceSampler
from tests.test_integrations.utils import IdealOneHotModel


def n_possible_triplets(n_labels: int, n_instances: int) -> int:
    return n_labels * (n_labels - 1) * (n_instances - 1) * n_instances**2


def get_labels(n_cls: int, sz: int) -> Tensor:
    labels = []
    for i in range(n_cls):
        labels.extend([i] * sz)
    shuffle(labels)
    return tensor(labels)


@pytest.mark.parametrize("n_cls,cls_sz,n_labels,n_instances", [[100, 8, 2, 8], [50, 10, 10, 4]])
@pytest.mark.parametrize("bank_sz,bank_k", [[5, 3], [20, 5], [5, 1]])
def test_mining_with_memory(
    n_cls: int, cls_sz: int, n_labels: int, n_instances: int, bank_sz: int, bank_k: int
) -> None:
    feat_dim = 2 * n_cls

    labels = get_labels(n_cls=n_cls, sz=cls_sz)
    sampler = BalanceSampler(labels=labels.tolist(), n_labels=n_labels, n_instances=n_instances)

    miner = TripletMinerWithMemory(bank_size_in_batches=bank_sz, tri_expand_k=bank_k)

    # we use shifted one-hot encoding here as a hack to avoid collapses with the features
    # from one-hot initialised memory bank in miner
    model = IdealOneHotModel(emb_dim=feat_dim, shift=bank_k + (n_labels * n_instances + 5))

    n_epoch = 2
    for i_epoch in range(n_epoch):
        for i, ii_batch in enumerate(sampler):
            if len(ii_batch) != n_labels * n_instances:  # to drop_last
                continue

            labels_batch = labels[ii_batch]
            features_batch = model(labels_batch)

            anch, pos, neg, is_original_tri = miner.sample(labels=labels_batch, features=features_batch)
            assert len(anch) == len(pos) == len(neg)
            assert n_possible_triplets(n_labels, n_instances) == int(is_original_tri.sum())

            n_out_tri = len(anch)
            n_desired_tri = bank_k * n_possible_triplets(n_labels, n_instances)

            # during the 1st epoch we have no chance to filter out some of the irrelevant triplets
            # because memory bank is empty
            if (i_epoch == 0) or (bank_k == 1):
                assert n_out_tri == n_desired_tri, (n_out_tri, n_desired_tri)
            # but later some of the collisions are possible, but their amount should not be high
            else:
                assert n_out_tri > 0.9 * n_desired_tri, (n_out_tri, n_desired_tri)

            # we use the fact that our features are one-hot encoded labels
            labels_sampled_anchor = torch.argmax(anch, dim=1)
            labels_sampled_pos = torch.argmax(pos, dim=1)
            labels_sampled_neg = torch.argmax(neg, dim=1)

            assert (labels_sampled_anchor == labels_sampled_pos).all()
            assert (labels_sampled_anchor != labels_sampled_neg).all()
