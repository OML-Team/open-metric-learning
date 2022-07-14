import math
from collections import Counter
from operator import itemgetter
from random import randint, shuffle
from typing import Any, Dict, List, Set, Tuple, Union

import pytest

from oml.samplers.category_balance import CategoryBalanceBatchSampler
from oml.samplers.distinct_category_balance import DistinctCategoryBalanceBatchSampler

TLabalesMappingCLI = List[Tuple[List[int], Dict[int, int], int, int, int]]


def generate_valid_categories_labels(num: int, guarantee_enough_labels: bool = True) -> TLabalesMappingCLI:
    """This function generates some valid inputs for category sampler.

    Parameters:
    num:
        number of samples to generate

    Returns:
        Samples in the following order: (labels, label2category, n_categories, n_labels, n_instances)
    """
    labels_cli = []

    for _ in range(num):
        unique_labels_number = randint(35, 55)
        n_labels, n_instances = randint(3, 7), randint(5, 10)
        unique_labels = list(range(unique_labels_number))
        shuffle(unique_labels)
        label2category = {}
        idx = 0
        cat = 0
        while idx < unique_labels_number:
            if guarantee_enough_labels:
                new_idx = idx + randint(0, 5) + n_labels
                labels_subset = unique_labels[idx:new_idx]
                if len(labels_subset) < n_labels:
                    cat -= 1
            else:
                new_idx = idx + randint(2, n_labels - 1)
                labels_subset = unique_labels[idx:new_idx]
                # process last segment of labels
                if len(labels_subset) == 1:
                    cat -= 1
            label2category.update({label: cat for label in labels_subset})
            idx = new_idx
            cat += 1
        n_categories = randint(1, len(set(label2category.values())))
        labels = []
        for label in unique_labels:
            labels.extend([label] * randint(2, 20))
        shuffle(labels)
        labels_cli.append((labels, label2category, n_categories, n_labels, n_instances))
    return labels_cli


@pytest.fixture()
def input_for_category_balance_batch_sampler_few_labels() -> TLabalesMappingCLI:
    """Generate a list of valid inputs for category balanced batch sampler with few labels for some categories.

    Returns:
        Test data for sampler in the following order: (labels, label2category, n_categories, n_labels, n_instances)
    """
    # (julia-shenshina) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases = generate_valid_categories_labels(num_random_cases, guarantee_enough_labels=False)
    return input_cases


@pytest.fixture()
def input_for_category_balance_batch_sampler() -> TLabalesMappingCLI:
    """Generate a list of valid inputs for category balanced batch sampler.

    Returns:
        Test data for sampler in the following order: (labels, label2category, n_categories, n_labels, n_instances)
    """
    # (julia-shenshina) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases = generate_valid_categories_labels(num_random_cases, guarantee_enough_labels=True)
    return input_cases


def check_category_balance_batch_sampler_epoch(
    sampler: Union[CategoryBalanceBatchSampler, DistinctCategoryBalanceBatchSampler],
    labels: List[int],
    label2category: Dict[int, int],
    n_categories: int,
    n_labels: int,
    n_instances: int,
    resample_labels: bool,
    epoch_size: int,
) -> None:
    """
    Args:
        sampler: Sampler to test
        labels: List of labels
        label2category: label to category mapping
        n_categories: Number of categories in a batch
        n_labels: Number of labels in a batch
        n_instances: Number of instances for each label in a batch
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
        assert len(batch_categories) == n_categories
        # check that new sampled at most p labels at all the batches
        if resample_labels:
            assert len(set(batch_labels)) <= n_labels * n_categories
        else:
            assert len(set(batch_labels)) == n_labels * n_categories
        collected_labels.update(batch_labels)

        labels_counter = Counter(batch_labels)
        num_batch_labels = len(labels_counter)
        num_batch_samples = list(labels_counter.values())
        cur_batch_size = len(batch_labels)

        # batch-level invariants
        assert len(set(batch_ids)) >= 4, set(batch_ids)  # type: ignore

        assert num_batch_labels <= n_categories * n_labels, (num_batch_labels, n_categories, n_labels)
        if resample_labels:
            assert all(el >= n_instances for el in num_batch_samples)
        else:
            assert all(el == n_instances for el in num_batch_samples)
        assert cur_batch_size == n_categories * n_labels * n_instances

    # epoch-level invariants
    assert len(collected_categories) <= len(set(label2category.values()))
    assert len(sampler) == epoch_size


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
    n_categories, n_labels, n_instances = 2, 3, 2
    with pytest.raises(ValueError, match="All the categories must have at least 3 unique labels"):
        _ = sampler_class(
            labels=labels,
            label2category=label2category,
            n_categories=n_categories,
            n_labels=n_labels,
            n_instances=n_instances,
            **sampler_kwargs
        )


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
    for labels, label2category, n_categories, n_labels, n_instances in fixture:
        sampler = CategoryBalanceBatchSampler(
            labels=labels,
            label2category=label2category,
            resample_labels=resample_labels,
            n_categories=n_categories,
            n_labels=n_labels,
            n_instances=n_instances,
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            n_categories=n_categories,
            n_labels=n_labels,
            n_instances=n_instances,
            resample_labels=resample_labels,
            epoch_size=math.ceil(len(set(labels)) / n_labels),
        )


def test_category_balance_batch_sampler_policy(input_for_category_balance_batch_sampler: TLabalesMappingCLI) -> None:
    """Check that CategoryBalanceBatchSampler behaves the same in case
    of valid input data with enough labels for all the categories for resample_labels "resample" and "raise".

    Args:
        input_for_category_balance_batch_sampler: Tuple of labels, label2category, n_categories, n_labels, n_instances)
    """
    for labels, label2category, n_categories, n_labels, n_instances in input_for_category_balance_batch_sampler:
        sampler = CategoryBalanceBatchSampler(
            labels=labels,
            label2category=label2category,
            resample_labels=True,
            n_categories=n_categories,
            n_labels=n_labels,
            n_instances=n_instances,
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            n_categories=n_categories,
            n_labels=n_labels,
            n_instances=n_instances,
            resample_labels=False,
            epoch_size=math.ceil(len(set(labels)) / n_labels),
        )


@pytest.mark.parametrize(
    "fixture_name,epoch_size",
    (
        # valid input: enough categories and labels for sampling without repetition
        ("input_for_category_balance_batch_sampler", 100),
    ),
)
def test_distinct_category_balance_batch_sampler(
    request: pytest.FixtureRequest, fixture_name: str, epoch_size: int
) -> None:
    """Check DistinctCategoryBalanceBatchSampler's behavior.

    Args:
        request: a request object to access to fixtures
        fixture_name: name of fixture to use
        epoch_size: number of batches in epoch
    """
    fixture = request.getfixturevalue(fixture_name)
    for labels, label2category, n_categories, n_labels, n_instances in fixture:
        sampler = DistinctCategoryBalanceBatchSampler(
            labels=labels,
            label2category=label2category,
            epoch_size=epoch_size,
            n_categories=n_categories,
            n_labels=n_labels,
            n_instances=n_instances,
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            n_categories=n_categories,
            n_labels=n_labels,
            n_instances=n_instances,
            resample_labels=True,
            epoch_size=epoch_size,
        )
