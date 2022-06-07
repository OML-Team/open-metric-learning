import pytest

from collections import Counter
from operator import itemgetter
from random import randint, shuffle
from typing import List, Tuple, Dict

from oml.samplers.balanced import BalanceBatchSampler, CategoryBalanceBatchSampler, Sampler, SequentialCategoryBalanceSampler

TLabelsPK = List[Tuple[List[int], int, int]]
TLabalesMappingCPK = List[Tuple[List[int], Dict[int, int], int, int, int]]


def generate_valid_labels(num: int) -> TLabelsPK:
    """
    This function generates some valid inputs for samplers.
    It generates k instances for p labels.

    Args:
        num: Number of generated samples

    Returns:
        Samples in the following order: (labels, p, k)

    """
    labels_pk = []

    for _ in range(num):
        p, k = randint(2, 12), randint(2, 12)
        labels_list = [[label] * randint(2, 12) for label in range(p)]
        labels = [el for sublist in labels_list for el in sublist]

        shuffle(labels)
        labels_pk.append((labels, p, k))

    return labels_pk


def generate_valid_categories_labels(num: int):
    """This function generates some valid inputs for category sampler.

    Parameters:
    num:
        number of samples to generate

    Returns:
        Samples in the following order: (labels, label2category, c, p, k)
    """
    labels_cpk = []

    for _ in range(num):
        unique_labels_number = randint(35, 55)
        p, k = randint(3, 7), randint(5, 10)
        unique_labels = list(range(unique_labels_number))
        shuffle(unique_labels)
        label2category = {}
        idx = 0
        cat = 0
        while idx < unique_labels_number:
            new_idx = idx + randint(0, 5) + p
            label2category.update({label: cat for label in unique_labels[idx: new_idx]})
            idx = new_idx
        c = randint(1, len(set(label2category.values())))
        labels = []
        for label in unique_labels:
            labels.extend([label] * randint(2, 20))
        shuffle(labels)
        labels_cpk.append((labels, label2category, c, p, k))
    return labels_cpk


@pytest.fixture()
def input_for_balance_batch_sampler() -> TLabelsPK:
    """
    Returns:
        Test data for sampler in the following order: (labels, p, k)

    """
    input_cases = [
        # ideal case
        ([0, 1, 2, 3, 0, 1, 2, 3], 2, 2),
        # repetation sampling is needed for label #3
        ([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], 2, 3),
        # check last batch behaviour:
        # last batch includes less than p labels (2 < 3)
        ([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], 3, 2),
        # we need to drop 1 label during the epoch because
        # number of labels in data % p = 1
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


@pytest.fixture()
def input_for_category_balance_batch_sampler() -> TLabalesMappingCPK:
    """Generate a list of valid inputs for category balanced batch sampler.

    Returns:
        Test data for sampler in the following order: (labels, label2category, c, p, k)
    """
    # (julia-shenshina) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases = generate_valid_categories_labels(num_random_cases)
    return input_cases


def check_balance_batch_sampler_epoch(sampler: Sampler, labels: List[int], p: int, k: int) -> None:
    """
    Args:
        sampler: Sampler to test
        labels: List of labels labels
        p: Number of labels in a batch
        k: Number of instances for each label in a batch

    """
    sampled_ids = list(sampler)

    sampled_labels = []
    collected_labels = set()
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
            assert 1 < num_batch_labels <= p
            assert all(1 < el <= k for el in num_batch_samples)
            assert 2 * 2 <= cur_batch_size <= p * k
        else:
            assert num_batch_labels == p, (num_batch_labels, p)
            assert all(el == k for el in num_batch_samples)
            assert cur_batch_size == p * k

    # epoch-level invariants
    num_labels_in_data = len(set(labels))
    num_labels_in_sampler = len(set(sampled_labels))
    assert (num_labels_in_data == num_labels_in_sampler) or (num_labels_in_data == num_labels_in_sampler + 1)

    n_instances_sampled = sum(map(len, sampled_ids))  # type: ignore
    assert (num_labels_in_data - 1) * k <= n_instances_sampled <= num_labels_in_data * k, (
        n_instances_sampled,
        num_labels_in_data * k,
    )


def check_category_balance_batch_sampler_epoch(
        sampler: Sampler, labels: List[int],
        label2category: Dict[int, int],
        c: int,
        p: int,
        k: int
):
    """
    Args:
        sampler: Sampler to test
        labels: List of labels labels
        label2category: label to category mapping
        c: Number of categories in a batch
        p: Number of labels in a batch
        k: Number of instances for each label in a batch

    """
    sampled_ids = list(sampler)

    collected_labels = set()
    # emulating of 1 epoch
    for i, batch_ids in enumerate(sampled_ids):
        batch_labels = itemgetter(*batch_ids)(labels)  # type: ignore
        batch_categories = set(label2category[label] for label in labels)
        # check that we sampled exactly c categories at all the batches
        assert len(batch_categories) == c
        # check that new batch collects at least one new label
        assert len(set(batch_labels) - collected_labels)
        collected_labels.update(batch_labels)

        labels_counter = Counter(batch_labels)
        num_batch_labels = len(labels_counter)
        num_batch_samples = list(labels_counter.values())
        cur_batch_size = len(batch_labels)

        # batch-level invariants
        assert len(set(batch_ids)) >= 4, set(batch_ids)  # type: ignore

        assert num_batch_labels == c * p, (num_batch_labels, c, p)
        assert all(el == k for el in num_batch_samples)
        assert cur_batch_size == c * p * k

    # epoch-level invariants
    num_labels_in_data = len(set(labels))
    num_labels_in_sampler = len(collected_labels)
    assert 0 <= num_labels_in_data - num_labels_in_sampler <= 1


def test_balance_batch_sampler(input_for_balance_batch_sampler):
    """
    Args:
        input_for_balance_batch_sampler: List of (labels, p, k)

    """
    for labels, p, k in input_for_balance_batch_sampler:
        sampler = BalanceBatchSampler(labels=labels, p=p, k=k)
        check_balance_batch_sampler_epoch(sampler=sampler, labels=labels, p=p, k=k)


def test_category_balance_batch_sampler(input_for_category_balance_batch_sampler):
    """Check that CategoryBalanceBatchSampler behaves the same with BalanceBatchSampler in case
    of the only category.

    Args:
        input_for_category_balance_batch_sampler: List of (labels, label2category, c, p, k)
    """
    for labels, label2category, c, p, k in input_for_category_balance_batch_sampler:
        sampler = CategoryBalanceBatchSampler(
            labels=labels, label2category=label2category, c=c, p=p, k=k
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler, labels=labels, label2category=label2category, c=c, p=p, k=k
        )


def test_sequential_category_balanced_batch_sampler(input_for_category_balance_batch_sampler):
    """
    Check if SequentialCategoryBalanceSampler __len__ method returns the real len of
    __iter__ output.
    """
    for labels, label2category, c, p, k in input_for_category_balance_batch_sampler:
        seq_sampler = SequentialCategoryBalanceSampler(
            labels=labels, label2category=label2category, c=c, p=p, k=k
        )
        indices = list(seq_sampler)
        assert len(indices) == len(seq_sampler)
