from collections import defaultdict
from itertools import chain
from math import ceil
from pprint import pprint
from random import randint

import pytest
from torch.utils.data import DataLoader

from oml.samplers.balance import BalanceSampler
from oml.utils.ddp import patch_dataloader_to_ddp, sync_dicts_ddp

from .utils import func_in_ddp, init_ddp


@pytest.mark.parametrize("n_labels_sampler", [2, 5])
@pytest.mark.parametrize("n_instances_sampler", [2, 5])
@pytest.mark.parametrize("n_labels_dataset", [100, 85])
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_patching_balance_batch_sampler(
    world_size: int, n_labels_dataset: int, n_labels_sampler: int, n_instances_sampler: int
) -> None:
    args = (n_labels_dataset, n_labels_sampler, n_instances_sampler)
    func_in_ddp(world_size=world_size, fn=check_patching_balance_batch_sampler, args=args)


def check_patching_balance_batch_sampler(
    rank: int, world_size: int, n_labels_dataset: int, n_labels_sampler: int, n_instances_sampler: int
) -> None:
    init_ddp(rank, world_size)
    labels = [[label] * randint(n_instances_sampler, 2 * n_instances_sampler) for label in range(n_labels_dataset)]
    labels = list(chain(*labels))
    dataset = list(range(len(labels)))

    batch_sampler = BalanceSampler(labels=labels, n_labels=n_labels_sampler, n_instances=n_instances_sampler)
    loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=lambda x: tuple(x))

    loader_ddp = patch_dataloader_to_ddp(loader)

    outputs_from_epochs = []

    for epoch in range(3):
        outputs = []
        for batch in loader_ddp:
            outputs.append(batch)

        outputs_from_epochs.append(outputs)

        seq_outputs = list(chain(*outputs))

        expected_len_per_process = ceil(len(loader) / max(1, world_size))

        assert len(outputs) == expected_len_per_process
        assert len(set(seq_outputs)) == len(seq_outputs)
        assert len(set(seq_outputs).intersection(set(dataset))) == len(seq_outputs)

        outputs_synced = sync_dicts_ddp({"batches": outputs}, world_size)["batches"]

        assert len(outputs_synced) == max(1, world_size) * expected_len_per_process

        _num_batches_after_ddp_padding = expected_len_per_process * max(1, world_size)
        possible_num_not_unique_batches = _num_batches_after_ddp_padding - len(loader)

        assert possible_num_not_unique_batches in list(range(world_size))
        assert len(outputs_synced) - len(set(outputs_synced)) <= possible_num_not_unique_batches

    assert all(outp != outputs_from_epochs[0] for outp in outputs_from_epochs[1:])


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
    loader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=lambda x: x,
    )

    loader_ddp = patch_dataloader_to_ddp(loader)

    outputs_from_epochs = []

    for epoch in range(2):
        outputs = []
        for batch in loader_ddp:
            outputs.extend(batch)

        outputs_from_epochs.append(outputs)

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

    if shuffle:
        assert all(outp != outputs_from_epochs[0] for outp in outputs_from_epochs[1:])
    else:
        assert all(outp == outputs_from_epochs[0] for outp in outputs_from_epochs[1:])
