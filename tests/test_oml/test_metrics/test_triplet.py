from collections import defaultdict
from functools import partial
from typing import Any

import pytest
import torch

from oml.const import CATEGORIES_KEY, EMBEDDINGS_KEY, OVERALL_CATEGORIES_KEY
from oml.metrics.triplets import AccuracyOnTriplets
from oml.utils.misc import one_hot

oh = partial(one_hot, dim=8)


@pytest.fixture()
def perfect_case() -> Any:
    batch = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(0), oh(1), oh(1), oh(1), oh(0)]),  # 1st and 2nd triplet
        CATEGORIES_KEY: ["cat", "dog"],  # 1st triplet  # 2nd triplet
    }

    gt_metrics = defaultdict(dict)  # type: ignore
    gt_metrics[OVERALL_CATEGORIES_KEY][AccuracyOnTriplets.metric_name] = 1
    gt_metrics["cat"][AccuracyOnTriplets.metric_name] = 1
    gt_metrics["dog"][AccuracyOnTriplets.metric_name] = 1

    return [batch], gt_metrics


@pytest.fixture()
def some_case() -> Any:
    batch1 = {
        EMBEDDINGS_KEY: torch.stack(
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
        CATEGORIES_KEY: [0, 1, 1, 1],  # triplets #1 #2 #3 #4
    }

    batch2 = {
        EMBEDDINGS_KEY: torch.stack(
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
        CATEGORIES_KEY: [0, 1],  # triplets #5 #6
    }

    gt_metrics = defaultdict(dict)  # type: ignore
    gt_metrics[OVERALL_CATEGORIES_KEY][AccuracyOnTriplets.metric_name] = 1 / 2
    gt_metrics[0][AccuracyOnTriplets.metric_name] = 1 / 2
    gt_metrics[1][AccuracyOnTriplets.metric_name] = 1 / 2

    return [batch1, batch2], gt_metrics


def run_accuracy_on_triplets(case) -> None:  # type: ignore
    batches, gt_metrics = case

    acc = AccuracyOnTriplets(embeddings_key=EMBEDDINGS_KEY, categories_key=CATEGORIES_KEY)
    acc.setup()
    for batch in batches:
        acc.update_data(batch)

    metrics = acc.compute_metrics()

    assert gt_metrics == metrics, (gt_metrics, metrics)


def test_perfect_case(perfect_case) -> None:  # type: ignore
    run_accuracy_on_triplets(perfect_case)


def test_some_case(some_case) -> None:  # type: ignore
    run_accuracy_on_triplets(some_case)
