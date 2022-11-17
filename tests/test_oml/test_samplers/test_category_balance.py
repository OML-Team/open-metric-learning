import math
from collections import Counter, defaultdict
from functools import partial
from operator import itemgetter
from random import randint, shuffle
from typing import Any, Dict, List, Set, Tuple, Union

import pytest

from oml.samplers.category_balance import CategoryBalanceSampler
from oml.samplers.distinct_category_balance import DistinctCategoryBalanceSampler
from oml.utils.misc import set_global_seed

TLabelsWithMapping = List[Tuple[List[int], Dict[int, int]]]


def generate_valid_categories_labels(num: int, guarantee_enough_labels: bool = True) -> TLabelsWithMapping:
    """
    This function generates some valid inputs for category sampler.

    Args:
        num: Number of samples to generate
        guarantee_enough_labels: If you want to guarantee enough amount of labels

    Returns:
        Labels with the mapping from labels to categories

    """
    labels_with_mapping: TLabelsWithMapping = []

    for _ in range(num):
        unique_labels_number = randint(35, 55)
        n_labels, _ = randint(3, 7), randint(5, 10)
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
        labels = []
        for label in unique_labels:
            labels.extend([label] * randint(2, 20))
        shuffle(labels)
        labels_with_mapping.append((labels, label2category))
    return labels_with_mapping


def get_valid_batch_params(labels: List[int], label2category: Dict[int, int]) -> Tuple[int, int, int]:
    # labels
    category2label = defaultdict(list)
    for label, category in label2category.items():
        category2label[category].append(label)
    categories_sizes = list(map(len, category2label.values()))
    n_labels = randint(2, min(categories_sizes))

    # categories
    n_categories = randint(1, len(set(label2category.values())))

    # instances
    n_instances = randint(2, max(Counter(labels).values()))

    return n_labels, n_instances, n_categories


@pytest.fixture()
def input_for_category_balance_batch_sampler_few_labels() -> TLabelsWithMapping:
    # (julia-shenshina) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases = generate_valid_categories_labels(num_random_cases, guarantee_enough_labels=False)
    return input_cases


@pytest.fixture()
def input_for_category_balance_batch_sampler() -> TLabelsWithMapping:
    # (julia-shenshina) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases = generate_valid_categories_labels(num_random_cases, guarantee_enough_labels=True)
    return input_cases


def check_category_balance_batch_sampler_epoch(
    sampler: Union[CategoryBalanceSampler, DistinctCategoryBalanceSampler],
    labels: List[int],
    label2category: Dict[int, int],
    resample_labels: bool,
    epoch_size: int,
) -> None:
    sampled_ids = list(sampler)

    collected_labels: Set[int] = set()
    collected_categories: Set[int] = set()

    # emulating of 1 epoch
    for i, batch_ids in enumerate(sampled_ids):
        assert len(batch_ids) == sampler.n_labels * sampler.n_categories * sampler.n_instances

        batch_labels = itemgetter(*batch_ids)(labels)  # type: ignore
        batch_categories = set(label2category[label] for label in batch_labels)

        # check that we sampled n_categories at all the batches
        assert len(batch_categories) == sampler.n_categories

        # check that new sampled at most n_labels at all the batches
        if resample_labels:
            assert len(set(batch_labels)) <= sampler.n_labels * sampler.n_categories
        else:
            assert len(set(batch_labels)) == sampler.n_labels * sampler.n_categories

        collected_labels.update(batch_labels)
        collected_categories.update(batch_categories)

        labels_counter = Counter(batch_labels)
        num_batch_labels = len(labels_counter)
        num_batch_samples = list(labels_counter.values())
        cur_batch_size = len(batch_labels)

        # batch-level invariants
        assert len(set(batch_ids)) >= 4, set(batch_ids)  # type: ignore

        assert num_batch_labels <= sampler.n_categories * sampler.n_labels, (
            num_batch_labels,
            sampler.n_categories,
            sampler.n_labels,
        )

        if resample_labels:
            assert all(el >= sampler.n_instances for el in num_batch_samples)
        else:
            assert all(el == sampler.n_instances for el in num_batch_samples)

        assert cur_batch_size == sampler.n_categories * sampler.n_labels * sampler.n_instances

    # epoch-level invariants
    assert len(collected_categories) <= len(set(label2category.values()))
    assert len(sampler) == epoch_size


@pytest.mark.parametrize(
    "sampler_class,sampler_kwargs",
    (
        (CategoryBalanceSampler, {"resample_labels": False}),
        (DistinctCategoryBalanceSampler, {"epoch_size": 20}),
    ),
)
def test_category_batch_sampler_resample_raises(sampler_class: Any, sampler_kwargs: Dict[str, Any]) -> None:
    """
    Check the behavior of samplers in case of lack of labels in a category.

    """
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

    assert True


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
    for labels, label2category in request.getfixturevalue(fixture_name):
        n_labels, n_instances, n_categories = get_valid_batch_params(labels, label2category)

        sampler = CategoryBalanceSampler(
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
            resample_labels=resample_labels,
            epoch_size=math.ceil(len(set(labels)) / n_labels),
        )

    assert True


def test_category_balance_batch_sampler_policy(input_for_category_balance_batch_sampler: TLabelsWithMapping) -> None:
    """
    Check that sampling with categories behaves the same in case
    of valid input data with enough labels for all the categories for resample_labels "resample" and "raise".

    """
    for labels, label2category in input_for_category_balance_batch_sampler:
        n_labels, n_instances, n_categories = get_valid_batch_params(labels, label2category)

        sampler = CategoryBalanceSampler(
            labels=labels,
            label2category=label2category,  # type: ignore
            n_labels=n_labels,
            n_instances=n_instances,
            n_categories=n_categories,
            resample_labels=True,
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            resample_labels=False,
            epoch_size=math.ceil(len(set(labels)) / n_labels),
        )

    assert True


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
    fixture = request.getfixturevalue(fixture_name)
    for labels, label2category in fixture:
        n_labels, n_instances, n_categories = get_valid_batch_params(labels, label2category)

        sampler = DistinctCategoryBalanceSampler(
            labels=labels,
            label2category=label2category,
            epoch_size=epoch_size,
            n_labels=n_labels,
            n_instances=n_instances,
            n_categories=n_categories,
        )
        check_category_balance_batch_sampler_epoch(
            sampler=sampler,
            labels=labels,
            label2category=label2category,
            resample_labels=True,
            epoch_size=epoch_size,
        )

    assert True


@pytest.mark.parametrize(
    "sampler_constructor",
    [CategoryBalanceSampler, partial(DistinctCategoryBalanceSampler, epoch_size=50)],
)
def test_categories_as_strings(sampler_constructor) -> None:  # type: ignore
    labels, label2category = generate_valid_categories_labels(1, guarantee_enough_labels=True)[0]

    n_labels, n_instances, n_categories = get_valid_batch_params(labels, label2category)

    label2category_str = {label: str(cat) for label, cat in label2category.items()}

    set_global_seed(0)
    sampler = sampler_constructor(labels, label2category, n_categories, n_labels, n_instances)
    ii_sampled = list(iter(sampler))

    set_global_seed(0)
    sampler_with_str = sampler_constructor(labels, label2category_str, n_categories, n_labels, n_instances)
    ii_sampled_with_str = list(iter(sampler_with_str))

    assert ii_sampled == ii_sampled_with_str, list(zip(ii_sampled, ii_sampled_with_str))
