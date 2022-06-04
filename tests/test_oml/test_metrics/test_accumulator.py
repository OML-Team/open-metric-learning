from typing import Any

import numpy as np
import pytest
import torch

from oml.metrics.accumulation import Accumulator


def get_data(n: int) -> Any:
    data = {"numpy": np.arange(n), "torch": torch.randn(n, 3), "list": list(range(n))}

    ii = list(range(0, n + 10, 10))
    data_as_batches = []
    for ia, ib in zip(ii, ii[1:]):
        batch = dict()
        for key in data.keys():
            batch[key] = data[key][ia:ib]
        data_as_batches.append(batch)

    return data, data_as_batches


@pytest.mark.parametrize("n", [1, 45, 20])
def test_accumulator(n: int) -> None:
    data, data_as_batches = get_data(n)

    acc = Accumulator(keys_to_accumulate=list(data.keys()))

    for _ in range(3):

        acc.refresh(num_samples=n)
        for batch in data_as_batches:
            acc.update_data(batch)

        assert all((acc.storage[k] == data[k]).all() for k in ["numpy", "torch"])
        assert all((acc.storage[k] == data[k]) for k in ["list"])
        assert acc.collected_samples == n
