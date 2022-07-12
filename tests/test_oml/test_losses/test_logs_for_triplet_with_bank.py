from math import isclose

import torch

from oml.losses.triplet import TripletLossWithMiner
from oml.miners.cross_batch import TripletMinerWithMemory
from tests.test_oml.test_miners.shared_checkers import calc_n_triplets


def test() -> None:
    expand_k = 2
    miner = TripletMinerWithMemory(bank_size_in_batches=3, tri_expand_k=expand_k)
    criterion = TripletLossWithMiner(margin=0.1, miner=miner, need_logs=True)

    labels = [0] * 4 + [1] * 3 + [2] * 6 + [3] * 2
    features = torch.rand(size=(len(labels), 16))

    _ = criterion(features=features, labels=torch.tensor(labels))
    logs = criterion.last_logs

    n_tri_orig = calc_n_triplets(labels=labels)
    n_tri_bank = n_tri_orig * (expand_k - 1)
    n_tri = n_tri_orig + n_tri_bank

    wo = n_tri_orig / n_tri
    wb = n_tri_bank / n_tri

    assert isclose(wb * logs["bank_active_tri"] + wo * logs["orig_active_tri"], logs["active_tri"], abs_tol=1e-3)
    assert isclose(wb * logs["pos_dist_bank"] + wo * logs["pos_dist_orig"], logs["pos_dist"], abs_tol=1e-3)
    assert isclose(wb * logs["neg_dist_bank"] + wo * logs["neg_dist_orig"], logs["neg_dist"], abs_tol=1e-3)
