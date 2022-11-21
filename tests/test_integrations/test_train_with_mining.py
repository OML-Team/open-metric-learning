import math
from random import randint, shuffle
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from oml.const import INPUT_TENSORS_KEY, LABELS_KEY
from oml.interfaces.datasets import IDatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.cross_batch import TripletMinerWithMemory
from oml.registry.miners import get_miner
from oml.samplers.balance import BalanceSampler
from tests.test_integrations.utils import IdealOneHotModel


class DummyDataset(IDatasetWithLabels):
    def __init__(self, n_labels: int, n_samples_min: int):
        self.labels = []
        for i in range(n_labels):
            self.labels.extend([i] * randint(n_samples_min, 2 * n_samples_min))
        shuffle(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {INPUT_TENSORS_KEY: torch.tensor(self.labels[idx]), LABELS_KEY: self.labels[idx]}

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> np.ndarray:
        return np.array(self.labels)


@pytest.mark.parametrize(
    "miner_name,miner_params",
    [
        ("hard_triplets", dict()),
        ("all_triplets", dict()),
        ("hard_cluster", dict()),
        ("triplets_with_memory", {"bank_size_in_batches": 5, "tri_expand_k": 3}),
    ],
)
@pytest.mark.parametrize("margin", [None, 0.5])
@pytest.mark.parametrize("n_labels,n_instances", [(2, 2), (5, 6)])
def test_train_with_mining(
    miner_name: str,
    miner_params: Dict[str, Any],
    margin: Optional[float],
    n_labels: int,
    n_instances: int,
) -> None:
    n_labels_total = n_labels * 6  # just some random figures

    dataset = DummyDataset(n_labels=n_labels_total, n_samples_min=n_instances)

    model = IdealOneHotModel(emb_dim=n_labels_total + randint(1, 5))

    loader = DataLoader(
        dataset=dataset,
        batch_sampler=BalanceSampler(labels=dataset.get_labels(), n_labels=n_labels, n_instances=n_instances),
    )

    miner = get_miner(miner_name, **miner_params)
    criterion = TripletLossWithMiner(margin=margin, miner=miner, need_logs=False)

    for i, batch in enumerate(loader):
        assert len(batch[LABELS_KEY]) == n_labels * n_instances

        embeddings = model(batch[INPUT_TENSORS_KEY])
        loss = criterion(embeddings, batch[LABELS_KEY])

        if isinstance(miner, TripletMinerWithMemory) and (i < miner.bank_size_in_batches):
            # we cannot guarantee any values of loss due to impact of memory bank initialisation
            continue
        else:
            if margin is None:
                # soft triplet loss
                assert loss.isclose(torch.log1p(torch.exp(torch.tensor(-math.sqrt(2)))), atol=1e-6)
            else:
                assert loss.isclose(torch.tensor(0.0))
