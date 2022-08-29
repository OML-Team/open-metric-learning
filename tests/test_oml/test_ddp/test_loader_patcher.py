from math import ceil

import pytest
from torch.utils.data import DataLoader

from oml.utils.ddp import patch_dataloader_to_ddp, sync_dicts_ddp

from .utils import func_in_ddp, init_ddp


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("batch_size", [10, 17])
@pytest.mark.parametrize("num_samples", [100, 85])
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_patching_seq_sampler(
    world_size: int, num_samples: int, drop_last: bool, shuffle: bool, batch_size: int, num_workers: int
) -> None:
    args = (num_samples, drop_last, shuffle, batch_size, num_workers)
    func_in_ddp(world_size=world_size, fn=check_patching_seq_sampler, args=args)


def check_patching_seq_sampler(
    rank: int, world_size: int, num_samples: int, drop_last: bool, shuffle: bool, batch_size: int, num_workers: int
) -> None:
    init_ddp(rank, world_size)
    dataset = list(range(num_samples))
    loader_no_ddp = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=lambda x: x,
    )

    loader_ddp = patch_dataloader_to_ddp(loader_no_ddp)

    outputs = []
    for batch in loader_ddp:
        outputs.extend(batch)

    if world_size > 1:
        expected_len_per_process = ceil(num_samples / world_size)
        if drop_last:
            expected_len_per_process = expected_len_per_process // batch_size * batch_size
    else:
        expected_len_per_process = num_samples // batch_size * batch_size if drop_last else num_samples

    assert len(outputs) == len(set(outputs)) == expected_len_per_process
    assert len(set(outputs).intersection(set(dataset))) == expected_len_per_process

    outputs_synced = sync_dicts_ddp({"ids": outputs}, world_size)["ids"]

    assert len(outputs_synced) == max(1, world_size) * expected_len_per_process

    _num_samples_after_ddp_padding = ceil(num_samples / max(1, world_size)) * max(1, world_size)
    possible_num_not_unique_samples = _num_samples_after_ddp_padding - num_samples

    assert possible_num_not_unique_samples <= max(1, world_size)
    assert len(outputs_synced) - len(set(outputs_synced)) <= possible_num_not_unique_samples
