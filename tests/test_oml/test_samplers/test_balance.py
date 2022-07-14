from collections import Counter
from operator import itemgetter
from random import randint, shuffle
from typing import List, Set, Tuple

import pytest

from oml.samplers.balance import BalanceBatchSampler

TLabelsPK = List[Tuple[List[int], int, int]]


def generate_valid_labels(num: int) -> TLabelsPK:
    """
    This function generates some valid inputs for samplers.
    It generates n_instances for n_labels.

    Args:
        num: Number of generated samples

    Returns:
        Samples in the following order: (labels, n_labels, n_instances)

    """
    labels_generated = []

    for _ in range(num):
        n_labels, n_instances = randint(2, 12), randint(2, 12)
        labels_list = [[label] * randint(2, 12) for label in range(n_labels)]
        labels = [el for sublist in labels_list for el in sublist]

        shuffle(labels)
        labels_generated.append((labels, n_labels, n_instances))

    return labels_generated


@pytest.fixture()
def input_for_balance_batch_sampler() -> TLabelsPK:
    """
    Returns:
        Test data for sampler in the following order: (labels, n_labels, n_instances)

    """
    input_cases = [
        # ideal case
        ([0, 1, 2, 3, 0, 1, 2, 3], 2, 2),
        # repetation sampling is needed for label #3
        ([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], 2, 3),
        # check last batch behaviour:
        # last batch includes less than n_labels (2 < 3)
        ([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], 3, 2),
        # we need to drop 1 label during the epoch because
        # number of labels in data % n_labels = 1
        ([0, 1, 2, 3, 0, 1, 2, 3], 3, 2),
        # several random cases
        ([0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2], 3, 5),
        ([0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2], 2, 3),
        ([0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2], 3, 2),
    ]

    # (alekseysh) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases.extend((generate_valid_labels(num_random_cases)))

    return input_cases


def check_balance_batch_sampler_epoch(
    sampler: BalanceBatchSampler, labels: List[int], n_labels: int, n_instances: int
) -> None:
    """
    Args:
        sampler: Sampler to test
        labels: List of labels labels
        n_labels: Number of labels in a batch
        n_instances: Number of instances for each label in a batch

    """
    sampled_ids = list(sampler)

    sampled_labels = []
    collected_labels: Set[int] = set()
    # emulating of 1 epoch
    for i, batch_ids in enumerate(sampled_ids):
        batch_labels = itemgetter(*batch_ids)(labels)  # type: ignore
        assert all(label not in collected_labels for label in batch_labels)
        collected_labels.update(batch_labels)

        labels_counter = Counter(batch_labels)
        num_batch_labels = len(labels_counter)
        num_batch_samples = list(labels_counter.values())
        cur_batch_size = len(batch_labels)
        sampled_labels.extend(list(labels_counter.keys()))

        # batch-level invariants
        assert len(set(batch_ids)) >= 4, set(batch_ids)  # type: ignore

        is_last_batch = i == sampler.batches_in_epoch - 1
        if is_last_batch:
            assert 1 < num_batch_labels <= n_labels
            assert all(1 < el <= n_instances for el in num_batch_samples)
            assert 2 * 2 <= cur_batch_size <= n_labels * n_instances
        else:
            assert num_batch_labels == n_labels, (num_batch_labels, n_labels)
            assert all(el == n_instances for el in num_batch_samples)
            assert cur_batch_size == n_labels * n_instances

    # epoch-level invariants
    num_labels_in_data = len(set(labels))
    num_labels_in_sampler = len(set(sampled_labels))
    assert (num_labels_in_data == num_labels_in_sampler) or (num_labels_in_data == num_labels_in_sampler + 1)

    n_instances_sampled = sum(map(len, sampled_ids))  # type: ignore
    assert (num_labels_in_data - 1) * n_instances <= n_instances_sampled <= num_labels_in_data * n_instances, (
        n_instances_sampled,
        num_labels_in_data * n_instances,
    )


def test_balance_batch_sampler(input_for_balance_batch_sampler: TLabelsPK) -> None:
    """
    Args:
        input_for_balance_batch_sampler: List of (labels, n_labels, n_instances)

    """
    for labels, n_labels, n_instances in input_for_balance_batch_sampler:
        sampler = BalanceBatchSampler(labels=labels, n_labels=n_labels, n_instances=n_instances)
        check_balance_batch_sampler_epoch(sampler=sampler, labels=labels, n_labels=n_labels, n_instances=n_instances)
