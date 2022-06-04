from random import shuffle

import pytest
import torch
from torch import Tensor, tensor

from oml.miners.among_batches import TripletMinerWithMemory
from oml.samplers.balanced import BalanceBatchSampler
from tests.test_integrations.utils import IdealOneHotModel


def n_possible_triplets(p: int, k: int) -> int:
    return p * (p - 1) * (k - 1) * k**2


def get_labels(n_cls: int, sz: int) -> Tensor:
    labels = []
    for i in range(n_cls):
        labels.extend([i] * sz)
    shuffle(labels)
    return tensor(labels)


@pytest.mark.parametrize("n_cls,cls_sz,p,k", [[100, 8, 2, 8], [50, 10, 10, 4]])
@pytest.mark.parametrize("bank_sz,bank_k", [[5, 3], [20, 5], [5, 1]])
def test_mining_with_memory(n_cls: int, cls_sz: int, p: int, k: int, bank_sz: int, bank_k: int) -> None:
    feat_dim = 2 * n_cls

    labels = get_labels(n_cls=n_cls, sz=cls_sz)
    sampler = BalanceBatchSampler(labels=labels.tolist(), p=p, k=k)

    miner = TripletMinerWithMemory(bank_size_in_batches=bank_sz, tri_expand_k=bank_k)

    # we use shifted one-hot encoding here as a hack to avoid collapses with the features
    # from one-hot initialised memory bank in miner
    model = IdealOneHotModel(emb_dim=feat_dim, shift=bank_k + (p * k + 5))

    n_epoch = 2
    for i_epoch in range(n_epoch):
        for i, ii_batch in enumerate(sampler):
            if len(ii_batch) != p * k:  # to drop_last
                continue

            labels_batch = labels[ii_batch]
            features_batch = model(labels_batch)

            anch, pos, neg = miner.sample(labels=labels_batch, features=features_batch)
            assert len(anch) == len(pos) == len(neg)

            n_out_tri = len(anch)
            n_desired_tri = bank_k * n_possible_triplets(p, k)

            # during the 1st epoch we have no chance to filter out some of the irrelevant triplets
            # because memory bank is empty
            if i_epoch == 0:
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
