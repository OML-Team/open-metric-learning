from collections import Counter
from operator import itemgetter
from random import randint, shuffle
from typing import List, Set

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
        n_labels = randint(2, 30)
        labels_list = [[label] * randint(2, 10) for label in range(n_labels)]
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

        labels_counter = Counter(batch_labels)
        num_batch_labels = len(labels_counter)
        num_batch_samples_counts = list(labels_counter.values())

        # batch-level invariants
        assert all(label not in collected_labels for label in batch_labels)
        assert len(set(batch_ids)) >= 4, set(batch_ids)  # type: ignore
        assert len(batch_ids) == sampler.n_labels * sampler.n_instances
        assert num_batch_labels == sampler.n_labels, (num_batch_labels, sampler.n_labels)
        assert all(el == sampler.n_instances for el in num_batch_samples_counts)

        collected_labels.update(batch_labels)

    # epoch-level invariants
    n_expected_batches = len(set(labels)) // sampler.n_labels
    num_labels_in_data = len(set(labels))
    num_labels_sampled = len(collected_labels)
    n_instances_sampled = sum(map(len, sampled_ids))  # type: ignore

    assert len(sampler) == n_expected_batches, (len(sampler), n_expected_batches)
    assert (num_labels_in_data - sampler.n_labels + 1) <= num_labels_sampled <= num_labels_in_data
    assert len(sampler) * sampler.n_labels * sampler.n_instances == n_instances_sampled


def test_balance_batch_sampler(input_for_balance_batch_sampler: TLabels) -> None:
    for labels in input_for_balance_batch_sampler:
        n_labels_batch = randint(2, max(2, len(set(labels)) // 5))
        n_instances_batch = randint(2, max(Counter(labels).values()))
        sampler = BalanceSampler(labels=labels, n_labels=n_labels_batch, n_instances=n_instances_batch)
        check_balance_batch_sampler_epoch(sampler=sampler, labels=labels)
