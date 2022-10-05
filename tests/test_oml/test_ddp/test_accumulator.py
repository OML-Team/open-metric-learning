from itertools import chain

import numpy as np
import pytest
import torch

from oml.metrics.accumulation import Accumulator

from .utils import init_ddp, run_in_ddp


@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_ddp_accumulator(world_size: int, device: str) -> None:
    run_in_ddp(world_size=world_size, fn=check_ddp_accumulator, args=(device,))


def check_ddp_accumulator(rank: int, world_size: int, device: str) -> None:
    init_ddp(rank, world_size)

    value = rank + 1

    data = {
        "list": [value] * value,
        "tensor_1d": value * torch.ones(value, device=device),
        "tensor_3d": value * torch.ones((value, 2, 3), device=device),
        "numpy_1d": value * np.ones(value),
        "numpy_3d": value * np.ones((value, 4, 5)),
    }

    acc = Accumulator(keys_to_accumulate=list(data.keys()))
    acc.refresh(len(data["list"]))
    acc.update_data(data)

    synced_data = acc.sync().storage

    len_after_sync = sum(range(1, world_size + 1))

    assert len(synced_data["list"]) == len_after_sync

    assert synced_data["tensor_1d"].ndim == 1  # type: ignore
    assert synced_data["tensor_1d"].shape == (len_after_sync,)  # type: ignore

    assert synced_data["tensor_3d"].ndim == 3  # type: ignore
    assert synced_data["tensor_3d"].shape == (len_after_sync, 2, 3)  # type: ignore

    assert synced_data["numpy_1d"].ndim == 1  # type: ignore
    assert synced_data["numpy_1d"].shape == (len_after_sync,)  # type: ignore

    assert synced_data["numpy_3d"].ndim == 3  # type: ignore
    assert synced_data["numpy_3d"].shape == (len_after_sync, 4, 5)  # type: ignore

    # 'sync' doesn't guarantee ordered data, so we can only compare the data structure
    expected_list = list(chain(*[[v] * v for v in range(1, world_size + 1)]))
    assert list(sorted(synced_data["list"])) == list(sorted(expected_list))

    expected_tensor_1d = torch.cat([v * torch.ones(v) for v in range(1, world_size + 1)], dim=0)
    sorted_synced_array, _ = torch.sort(synced_data["tensor_1d"], dim=0)
    assert torch.all(torch.isclose(expected_tensor_1d, sorted_synced_array))

    expected_tensor_3d = torch.cat([v * torch.ones((v, 2, 3)) for v in range(1, world_size + 1)], dim=0)
    sorted_synced_array, _ = torch.sort(synced_data["tensor_3d"], dim=0)
    assert torch.all(torch.isclose(expected_tensor_3d, sorted_synced_array))

    expected_numpy_1d = np.concatenate([v * np.ones(v) for v in range(1, world_size + 1)], axis=0)
    sorted_synced_array = np.sort(synced_data["numpy_1d"], axis=0)
    assert np.all(np.isclose(expected_numpy_1d, sorted_synced_array))

    expected_numpy_3d = np.concatenate([v * np.ones((v, 4, 5)) for v in range(1, world_size + 1)], axis=0)
    sorted_synced_array = np.sort(synced_data["numpy_3d"], axis=0)
    assert np.all(np.isclose(expected_numpy_3d, sorted_synced_array))
