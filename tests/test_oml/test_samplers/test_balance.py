from collections import Counter
from math import ceil
from operator import itemgetter
from random import randint, shuffle
from typing import List, Set, Tuple

import pytest

from oml.samplers.balance import BalanceSampler

TLabels = List[List[int]]


def generate_valid_labels(num: int) -> TLabels:
    """
    This function generates some valid inputs for samplers.
    It generates n_instances for n_labels.

    Args:
        num: Number of generated samples

    Returns:
        Labels

    """
    labels_generated = []

    for _ in range(num):
        n_labels = randint(2, 22)
        labels_list = [[label] * randint(2, 22) for label in range(n_labels)]
        labels = [el for sublist in labels_list for el in sublist]

        shuffle(labels)
        labels_generated.append(labels)

    return labels_generated


@pytest.fixture()
def input_for_balance_batch_sampler() -> TLabels:
    """
    Returns:
        Test data for sampler in the following order: (labels, n_labels, n_instances)

    """
    input_cases = [
        # ideal case
        [0, 1, 2, 3, 0, 1, 2, 3],
        # repetition sampling is needed for label #3
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
        # check last batch behaviour:
        # last batch includes less than n_labels (2 < 3)
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        # we need to drop 1 label during the epoch because
        # number of labels in data % n_labels = 1
        [0, 1, 2, 3, 0, 1, 2, 3],
        # several random cases
        [0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2],
        [0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2],
        [0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2],
    ]

    # (alekseysh) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases.extend((generate_valid_labels(num_random_cases)))

    return input_cases


def check_balance_batch_sampler_epoch(sampler: BalanceSampler, labels: List[int]) -> None:
    sampled_ids = list(sampler)

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

        # batch-level invariants
        assert len(set(batch_ids)) >= 4, set(batch_ids)  # type: ignore

        is_last_batch = i == len(sampler) - 1
        if is_last_batch:
            assert 1 < num_batch_labels <= sampler.n_labels
            assert all(1 < el <= sampler.n_instances for el in num_batch_samples)
            assert 2 * 2 <= cur_batch_size <= sampler.n_labels * sampler.n_instances
        else:
            assert num_batch_labels == sampler.n_labels, (num_batch_labels, sampler.n_labels)
            assert all(el == sampler.n_instances for el in num_batch_samples)
            assert cur_batch_size == sampler.n_labels * sampler.n_instances

    # epoch-level invariants
    n_expected_batches = ceil(len(set(labels)) / sampler.n_labels)
    num_labels_in_data = len(set(labels))
    num_labels_sampled = len(collected_labels)

    assert len(sampler) in [n_expected_batches, n_expected_batches - 1]
    assert num_labels_in_data in [num_labels_sampled, num_labels_sampled + 1]

    n_instances_sampled = sum(map(len, sampled_ids))  # type: ignore
    bs = sampler.n_labels * sampler.n_instances
    assert (len(sampler) - 1) * bs <= n_instances_sampled <= len(sampler) * bs


def test_balance_batch_sampler(input_for_balance_batch_sampler: TLabels) -> None:
    """
    Args:
        input_for_balance_batch_sampler: List of (labels, n_labels, n_instances)

    """
    for labels in input_for_balance_batch_sampler:
        n_labels_batch = randint(2, len(set(labels)))
        n_instances_batch = randint(2, max(Counter(labels).values()))
        sampler = BalanceSampler(labels=labels, n_labels=n_labels_batch, n_instances=n_instances_batch)
        check_balance_batch_sampler_epoch(sampler=sampler, labels=labels)
