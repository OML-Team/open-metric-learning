# type: ignore
from itertools import chain
from math import ceil
from random import randint
from typing import Any, Dict, List

import pytest
from torch.utils.data import DataLoader

from oml.ddp.patching import patch_dataloader_to_ddp
from oml.ddp.utils import sync_dicts_ddp
from oml.interfaces.samplers import IBatchSampler
from oml.samplers.balance import BalanceSampler
from oml.samplers.category_balance import CategoryBalanceSampler
from oml.samplers.distinct_category_balance import DistinctCategoryBalanceSampler

from .utils import init_ddp, run_in_ddp


@pytest.mark.parametrize("n_labels_sampler", [2, 5])
@pytest.mark.parametrize("n_instances_sampler", [2, 5])
@pytest.mark.parametrize("n_labels_dataset", [100, 85])
@pytest.mark.parametrize(
    "sampler_class, setup_kwargs",
    [
        (BalanceSampler, {}),
        (CategoryBalanceSampler, {"n_categories": 2, "num_categories_in_dataset": 10}),
        (CategoryBalanceSampler, {"n_categories": 3, "num_categories_in_dataset": 17}),
        (DistinctCategoryBalanceSampler, {"n_categories": 2, "num_categories_in_dataset": 10, "epoch_size": 4}),
        (DistinctCategoryBalanceSampler, {"n_categories": 4, "num_categories_in_dataset": 17, "epoch_size": 5}),
    ],
)
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_patching_balance_sampler(
    world_size: int,
    n_labels_dataset: int,
    n_labels_sampler: int,
    n_instances_sampler: int,
    sampler_class: IBatchSampler,
    setup_kwargs: Dict[str, Any],
) -> None:
    args = (n_labels_dataset, n_labels_sampler, n_instances_sampler, sampler_class, setup_kwargs)
    run_in_ddp(world_size=world_size, fn=check_patching_balance_batch_sampler, args=args)


def setup_batch_sampler(
    sampler_class: IBatchSampler, labels: List[int], n_labels: int, n_instances: int, **kwargs: Dict[str, Any]
) -> IBatchSampler:
    class2setup = {
        BalanceSampler: _setup_balance_sampler,
        CategoryBalanceSampler: _setup_category_sampler,
        DistinctCategoryBalanceSampler: _setup_distinct_category_sampler,
    }

    return class2setup[sampler_class](labels=labels, n_labels=n_labels, n_instances=n_instances, **kwargs)


def _setup_balance_sampler(
    labels: List[int], n_labels: int, n_instances: int, **kwargs: Dict[str, Any]
) -> BalanceSampler:
    return BalanceSampler(labels=labels, n_labels=n_labels, n_instances=n_instances)


def _setup_category_sampler(
    labels: List[int],
    n_labels: int,
    n_instances: int,
    n_categories: int,
    num_categories_in_dataset: int,
    **kwargs: Dict[str, Any]
) -> CategoryBalanceSampler:
    label2category = dict(zip(labels, [label % num_categories_in_dataset for label in labels]))
    return CategoryBalanceSampler(
        labels=labels,
        label2category=label2category,
        n_categories=n_categories,
        n_labels=n_labels,
        n_instances=n_instances,
    )


def _setup_distinct_category_sampler(
    labels: List[int],
    n_labels: int,
    n_instances: int,
    n_categories: int,
    num_categories_in_dataset: int,
    epoch_size: int,
    **kwargs: Dict[str, Any]
) -> DistinctCategoryBalanceSampler:
    label2category = dict(zip(labels, [label % num_categories_in_dataset for label in labels]))
    return DistinctCategoryBalanceSampler(
        labels=labels,
        label2category=label2category,
        n_categories=n_categories,
        n_labels=n_labels,
        n_instances=n_instances,
        epoch_size=epoch_size,
    )


def check_patching_balance_batch_sampler(
    rank: int,
    world_size: int,
    n_labels_dataset: int,
    n_labels_sampler: int,
    n_instances_sampler: int,
    sampler_class,
    setup_kwargs,
) -> None:
    init_ddp(rank, world_size)
    labels = [[label] * randint(n_instances_sampler, 2 * n_instances_sampler) for label in range(n_labels_dataset)]
    labels = list(chain(*labels))
    dataset = list(range(len(labels)))

    batch_sampler = setup_batch_sampler(
        sampler_class, labels=labels, n_labels=n_labels_sampler, n_instances=n_instances_sampler, **setup_kwargs
    )

    loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=lambda x: tuple(x))

    loader_ddp = patch_dataloader_to_ddp(loader)

    outputs_from_epochs = []

    for epoch in range(3):
        outputs = []
        for batch in loader_ddp:
            outputs.append(batch)

        outputs_from_epochs.append(outputs)

        seq_outputs = list(chain(*outputs))

        # number of batches should be divided by the number of devices with padding
        expected_len_per_process = ceil(len(loader) / max(1, world_size))
        assert len(outputs) == expected_len_per_process

        outputs_synced = sync_dicts_ddp({"batches": outputs}, world_size)["batches"]

        assert len(outputs_synced) == max(1, world_size) * expected_len_per_process

        if sampler_class == BalanceSampler:
            # Check each batches without repeating of ids
            assert len(set(seq_outputs)) == len(seq_outputs)

            _num_batches_after_ddp_padding = expected_len_per_process * max(1, world_size)
            possible_num_not_unique_batches = _num_batches_after_ddp_padding - len(loader)

            assert possible_num_not_unique_batches < world_size
            assert len(outputs_synced) - len(set(outputs_synced)) <= possible_num_not_unique_batches

    # we check that batches on each epoch are different
    outputs_from_epochs = list(map(tuple, outputs_from_epochs))
    assert len(set(outputs_from_epochs)) == len(outputs_from_epochs)


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("batch_size", [10, 17])
@pytest.mark.parametrize("num_samples", [100, 85])
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_patching_seq_sampler(
    world_size: int, num_samples: int, drop_last: bool, shuffle: bool, batch_size: int, num_workers: int
) -> None:
    args = (num_samples, drop_last, shuffle, batch_size, num_workers)
    run_in_ddp(world_size=world_size, fn=check_patching_seq_sampler, args=args)


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

        # check that ids are uniques and amount of them is reduced
        assert len(outputs) == len(set(outputs)) == expected_len_per_process

        outputs_synced = sync_dicts_ddp({"ids": outputs}, world_size)["ids"]

        assert len(outputs_synced) == max(1, world_size) * expected_len_per_process

        _num_samples_after_ddp_padding = ceil(num_samples / max(1, world_size)) * max(1, world_size)
        possible_num_not_unique_samples = _num_samples_after_ddp_padding - num_samples

        assert possible_num_not_unique_samples < world_size
        assert len(outputs_synced) - len(set(outputs_synced)) <= possible_num_not_unique_samples

    # Depends on the shuffle bathes should be the same or different
    outputs_from_epochs = list(map(tuple, outputs_from_epochs))
    if shuffle:
        assert len(set(outputs_from_epochs)) == len(outputs_from_epochs)
    else:
        assert len(set(outputs_from_epochs)) != len(outputs_from_epochs)
