import math
from collections import Counter
from operator import itemgetter
from random import randint, shuffle
from typing import Any, Dict, List, Set, Tuple, Union

import pytest

from oml.samplers.balanced import (
    BalanceBatchSampler,
    CategoryBalanceBatchSampler,
    DistinctCategoryBalanceBatchSampler,
)

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


def generate_valid_categories_labels(num: int, guarantee_enough_labels: bool = True) -> TLabalesMappingCPK:
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
            if guarantee_enough_labels:
                new_idx = idx + randint(0, 5) + p
                labels_subset = unique_labels[idx:new_idx]
                if len(labels_subset) < p:
                    cat -= 1
            else:
                new_idx = idx + randint(2, p - 1)
                labels_subset = unique_labels[idx:new_idx]
                # process last segment of labels
                if len(labels_subset) == 1:
                    cat -= 1
            label2category.update({label: cat for label in labels_subset})
            idx = new_idx
            cat += 1
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
    input_cases = generate_valid_categories_labels(num_random_cases, guarantee_enough_labels=True)
    return input_cases


@pytest.fixture()
def input_for_category_balance_batch_sampler_few_labels() -> TLabalesMappingCPK:
    """Generate a list of valid inputs for category balanced batch sampler with few labels for some categories.

    Returns:
        Test data for sampler in the following order: (labels, label2category, c, p, k)
    """
    # (julia-shenshina) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases = generate_valid_categories_labels(num_random_cases, guarantee_enough_labels=False)
    return input_cases


def check_balance_batch_sampler_epoch(sampler: BalanceBatchSampler, labels: List[int], p: int, k: int) -> None:
    """
    Args:
        sampler: Sampler to test
        labels: List of labels labels
        p: Number of labels in a batch
        k: Number of instances for each label in a batch

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
    sampler: Union[CategoryBalanceBatchSampler, DistinctCategoryBalanceBatchSampler],
    labels: List[int],
    label2category: Dict[int, int],
    c: int,
    p: int,
    k: int,
    resample_labels: bool,
    epoch_size: int,
) -> None:
    """
    Args:
        sampler: Sampler to test
        labels: List of labels
        label2category: label to category mapping
        c: Number of categories in a batch
        p: Number of labels in a batch
        k: Number of instances for each label in a batch
        resample_labels: resample strategy
        epoch_size: expected number of batches in epoch
    """
    sampled_ids = list(sampler)

    collected_labels: Set[int] = set()
    collected_categories: Set[int] = set()
    # emulating of 1 epoch
    for i, batch_ids in enumerate(sampled_ids):
        batch_labels = itemgetter(*batch_ids)(labels)  # type: ignore
        batch_categories = set(label2category[label] for label in batch_labels)
        # check that we sampled c categories at all the batches
        assert len(batch_categories) == c
        # check that new sampled at most p labels at all the batches
        if resample_labels:
            assert len(set(batch_labels)) <= p * c
        else:
            assert len(set(batch_labels)) == p * c
        collected_labels.update(batch_labels)

        labels_counter = Counter(batch_labels)
        num_batch_labels = len(labels_counter)
        num_batch_samples = list(labels_counter.values())
        cur_batch_size = len(batch_labels)

        # batch-level invariants
        assert len(set(batch_ids)) >= 4, set(batch_ids)  # type: ignore

        assert num_batch_labels <= c * p, (num_batch_labels, c, p)
        if resample_labels:
            assert all(el >= k for el in num_batch_samples)
        else:
            assert all(el == k for el in num_batch_samples)
        assert cur_batch_size == c * p * k

    # epoch-level invariants
    assert len(collected_categories) <= len(set(label2category.values()))
    assert len(sampler) == epoch_size


def test_balance_batch_sampler(input_for_balance_batch_sampler: TLabelsPK) -> None:
    """
    Args:
        input_for_balance_batch_sampler: List of (labels, p, k)

    """
    for labels, p, k in input_for_balance_batch_sampler:
        sampler = BalanceBatchSampler(labels=labels, p=p, k=k)
        check_balance_batch_sampler_epoch(sampler=sampler, labels=labels, p=p, k=k)


@pytest.mark.parametrize(
    "fixture_name,resample_labels",
    (
        # valid input: enough categories and labels for sampling without repetition
        ("input_for_category_balance_batch_sampler", False),
        ("input_for_category_balance_batch_sampler", True),
        # not enough labels in some categories
        ("input_for_category_balance_batch_sampler_few_labels", True),
    ),
)
def test_category_balance_batch_sampler(
    request: pytest.FixtureRequest, fixture_name: str, resample_labels: bool
) -> None:
    """Check CategoryBalanceBatchSampler's behavior.

    Args:
        request: a request object to access to fixtures
        fixture_name: name of fixture to use
        resample_labels: few labels resampling strategy
    """
    fixture = request.getfixturevalue(fixture_name)
    for labels, label2category, c, p, k in fixture:
        sampler = CategoryBalanceBatchSampler(
            labels=labels, label2category=label2category, c=c, p=p, k=k, resample_labels=resample_labels
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            c=c,
            p=p,
            k=k,
            resample_labels=resample_labels,
            epoch_size=math.ceil(len(set(labels)) / p),
        )


@pytest.mark.parametrize(
    "fixture_name,epoch_size",
    (
        # valid input: enough categories and labels for sampling without repetition
        ("input_for_category_balance_batch_sampler", 100),
    ),
)
def test_new_category_balance_batch_sampler(request: pytest.FixtureRequest, fixture_name: str, epoch_size: int) -> None:
    """Check DistinctCategoryBalanceBatchSampler's behavior.

    Args:
        request: a request object to access to fixtures
        fixture_name: name of fixture to use
        epoch_size: number of batches in epoch
    """
    fixture = request.getfixturevalue(fixture_name)
    for labels, label2category, c, p, k in fixture:
        sampler = DistinctCategoryBalanceBatchSampler(
            labels=labels, label2category=label2category, c=c, p=p, k=k, epoch_size=epoch_size
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            c=c,
            p=p,
            k=k,
            resample_labels=True,
            epoch_size=epoch_size,
        )


@pytest.mark.parametrize(
    "sampler_class,sampler_kwargs",
    (
        (CategoryBalanceBatchSampler, {"resample_labels": False}),
        (DistinctCategoryBalanceBatchSampler, {"epoch_size": 20}),
    ),
)
def test_category_batch_sampler_resample_raises(sampler_class: Any, sampler_kwargs: Dict[str, Any]) -> None:
    """Check the behavior of samplers in case of lack of labels in a category."""
    label2category = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    labels = [0] * 5 + [1] * 4 + [2] * 6 + [3] * 5 + [4] * 5 + [5] * 6
    c, p, k = 2, 3, 2
    with pytest.raises(ValueError, match="All the categories must have at least 3 unique labels"):
        _ = sampler_class(labels=labels, label2category=label2category, c=c, p=p, k=k, **sampler_kwargs)


def test_category_balance_batch_sampler_policy(input_for_category_balance_batch_sampler: TLabalesMappingCPK) -> None:
    """Check that CategoryBalanceBatchSampler behaves the same in case
    of valid input data with enough labels for all the categories for resample_labels "resample" and "raise".

    Args:
        input_for_category_balance_batch_sampler: Tuple of labels, label2category, c, p, k
    """
    for labels, label2category, c, p, k in input_for_category_balance_batch_sampler:
        sampler = CategoryBalanceBatchSampler(
            labels=labels, label2category=label2category, c=c, p=p, k=k, resample_labels=True
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            c=c,
            p=p,
            k=k,
            resample_labels=False,
            epoch_size=math.ceil(len(set(labels)) / p),
        )
