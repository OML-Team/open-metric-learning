from functools import partial
from typing import Any

import pytest
import torch

from oml.metrics.triplets import AccuracyOnTriplets
from oml.utils.misc import one_hot

oh = partial(one_hot, dim=8)


@pytest.fixture()
def perfect_case() -> Any:
    batch = {
        "embeddings": torch.stack([oh(0), oh(0), oh(1), oh(1), oh(1), oh(0)]),  # 1st and 2nd triplet
        "categories": ["cat", "dog"],  # 1st triplet  # 2nd triplet
    }

    categories_mapping = {"cat": 0, "dog": 1}

    gt_metrics = {
        f"{AccuracyOnTriplets.metric_name}/OVERALL": 1,
        f"{AccuracyOnTriplets.metric_name}/0": 1,
        f"{AccuracyOnTriplets.metric_name}/1": 1,
    }

    return [batch], gt_metrics, categories_mapping


@pytest.fixture()
def some_case() -> Any:
    batch1 = {
        "embeddings": torch.stack(
            [
                # triplet #1 - error
                oh(0),
                oh(5),
                oh(0),
                # triplet #2 - error
                oh(1),
                oh(0),
                oh(1),
                # triplet #3 - correct
                oh(5),
                oh(5),
                oh(1),
                # triplet #4 - correct
                oh(3),
                oh(3),
                oh(2),
            ]
        ),
        "categories": [0, 1, 1, 1],  # triplets #1 #2 #3 #4
    }

    batch2 = {
        "embeddings": torch.stack(
            [
                # triplet #5 - correct
                oh(0),
                oh(0),
                oh(1),
                # triplet #6 - error
                oh(1),
                oh(0),
                oh(1),
            ]
        ),
        "categories": [0, 1],  # triplets #5 #6
    }

    categories_mapping = {0: "cat", 1: "dog"}

    gt_metrics = {
        f"{AccuracyOnTriplets.metric_name}/OVERALL": 1 / 2,
        f"{AccuracyOnTriplets.metric_name}/cat": 1 / 2,
        f"{AccuracyOnTriplets.metric_name}/dog": 1 / 2,
    }

    return [batch1, batch2], gt_metrics, categories_mapping


def run_accuracy_on_triplets(case) -> None:  # type: ignore
    batches, gt_metrics, categories_mapping = case

    acc = AccuracyOnTriplets(
        embeddings_key="embeddings", categories_key="categories", categories_names_mapping=categories_mapping
    )
    acc.setup()
    for batch in batches:
        acc.update_data(batch)

    metrics = acc.compute_metrics()

    assert gt_metrics == metrics, (gt_metrics, metrics)


def test_perfect_case(perfect_case) -> None:  # type: ignore
    run_accuracy_on_triplets(perfect_case)


def test_some_case(some_case) -> None:  # type: ignore
    run_accuracy_on_triplets(some_case)
