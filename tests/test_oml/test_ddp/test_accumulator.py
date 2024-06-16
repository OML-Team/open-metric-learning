from itertools import chain

import numpy as np
import pytest
import torch

from oml.ddp.utils import get_rank_safe
from oml.metrics.accumulation import Accumulator

from .utils import run_in_ddp


@pytest.mark.long
@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
@pytest.mark.parametrize("create_duplicate", [True, False])
def test_ddp_accumulator(world_size: int, device: str, create_duplicate: bool) -> None:
    run_in_ddp(world_size=world_size, fn=check_accumulator, args=(world_size, device, create_duplicate))


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
@pytest.mark.parametrize("create_duplicate", [True, False])
def test_fake_ddp_accumulator(device: str, create_duplicate: bool) -> None:
    # we expect the same duplicate removing behaviour without initializing DDP
    check_accumulator(world_size=1, device=device, create_duplicate=create_duplicate)


def check_accumulator(world_size: int, device: str, create_duplicate: bool) -> None:
    rank = get_rank_safe()
    value = rank + 1
    size = value

    indices = {0: [0], 1: [1, 2], 2: [3, 4, 5]}[rank]

    if create_duplicate and (rank == 0):
        # let's pretend we doubled our single record at the rank 0
        size = 2
        indices = [0, 0]

    data = {
        "list": [value] * size,
        "tensor_1d": value * torch.ones(size, device=device),
        "tensor_3d": value * torch.ones((size, 2, 3), device=device),
        "numpy_1d": value * np.ones(size),
        "numpy_3d": value * np.ones((size, 4, 5)),
    }

    acc = Accumulator(keys_to_accumulate=tuple(data.keys()))
    acc.refresh(len(data["list"]))
    acc.update_data(data, indices=indices)

    acc_synced = acc.sync()
    synced_data = acc_synced.storage
    synced_num_samples = acc_synced.num_samples

    assert acc_synced.is_storage_full()

    len_after_sync = sum(range(1, world_size + 1))

    indices_synced = synced_data[acc._indices_key]

    assert len_after_sync == synced_num_samples

    assert len(indices_synced) == len(set(indices_synced))
    assert sorted(indices_synced) == list(range(len_after_sync))

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
