import math
from random import randint, shuffle
from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from oml.interfaces.datasets import IDatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.cross_batch import TripletMinerWithMemory
from oml.registry.miners import get_miner
from oml.samplers.balance import BalanceBatchSampler, SequentialBalanceSampler
from tests.test_integrations.utils import IdealOneHotModel


class DummyDataset(IDatasetWithLabels):
    def __init__(self, n_labels: int, n_samples_min: int):
        self.labels = []
        for i in range(n_labels):
            self.labels.extend([i] * randint(n_samples_min, 2 * n_samples_min))
        shuffle(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"input_tensors": torch.tensor(self.labels[idx]), "labels": self.labels[idx]}

    def __len__(self) -> int:
        return len(self.labels)

    def get_labels(self) -> np.ndarray:
        return np.array(self.labels)


@pytest.mark.parametrize("sampler_constructor", [BalanceBatchSampler, SequentialBalanceSampler])
@pytest.mark.parametrize(
    "miner_name,miner_params",
    [
        ("HardTripletsMiner", {"norm_required": True}),
        ("AllTripletsMiner", dict()),
        ("HardClusterMiner", dict()),
        ("TripletMinerWithMemory", {"bank_size_in_batches": 5, "tri_expand_k": 3}),
    ],
)
@pytest.mark.parametrize("margin", [None, 0.5])
@pytest.mark.parametrize("p,k", [(2, 2), (5, 6)])
def test_train_with_mining(sampler_constructor, miner_name, miner_params, margin, p, k) -> None:  # type: ignore
    n_labels_total = p * 6  # just some random figures

    dataset = DummyDataset(n_labels=n_labels_total, n_samples_min=k)

    model = IdealOneHotModel(emb_dim=n_labels_total + randint(1, 5))

    if sampler_constructor == BalanceBatchSampler:
        loader = DataLoader(dataset=dataset, batch_sampler=sampler_constructor(labels=dataset.get_labels(), p=p, k=k))
    elif sampler_constructor == SequentialBalanceSampler:
        loader = DataLoader(
            dataset=dataset, sampler=sampler_constructor(labels=dataset.get_labels(), p=p, k=k), batch_size=p * k
        )
    else:
        raise ValueError(f"Unexpected sampler: {sampler_constructor}.")

    miner = get_miner(miner_name, **miner_params)
    criterion = TripletLossWithMiner(margin=margin, miner=miner, need_logs=False)

    for i, batch in enumerate(loader):
        assert len(batch["labels"]) == p * k

        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])

        if isinstance(miner, TripletMinerWithMemory) and (i < miner.bank_size_in_batches):
            # we cannot guarantee any values of loss due to impact of memory bank initialisation
            continue
        else:
            if margin is None:
                # soft triplet loss
                assert loss.isclose(torch.log1p(torch.exp(torch.tensor(-math.sqrt(2)))), atol=1e-6)
            else:
                assert loss.isclose(torch.tensor(0.0))
