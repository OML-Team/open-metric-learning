import math
from collections import defaultdict
from functools import partial
from typing import Any

import pytest
import torch

from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import one_hot

oh = partial(one_hot, dim=8)


@pytest.fixture()
def perfect_case() -> Any:
    """
    Here we assume that our model provides the best possible embeddings:
    for the item with the label parameter equals to i,
    it provides one hot vector with non zero element in i-th position.

    Thus, we expect all of the metrics equals to 1.
    """

    batch1 = {
        "embeddings": torch.stack([oh(0), oh(1), oh(1)]),
        "labels": torch.tensor([0, 1, 1]),
        "is_query": torch.tensor([True, True, True]),
        "is_gallery": torch.tensor([False, False, False]),
        "categories": ["cat", "dog", "dog"],
    }

    batch2 = {
        "embeddings": torch.stack([oh(0), oh(1), oh(1)]),
        "labels": torch.tensor([0, 1, 1]),
        "is_query": torch.tensor([False, False, False]),
        "is_gallery": torch.tensor([True, True, True]),
        "categories": ["cat", "dog", "dog"],
    }

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics["OVERALL"]["cmc"][k] = 1.0
    metrics["cat"]["cmc"][k] = 1.0
    metrics["dog"]["cmc"][k] = 1.0

    return (batch1, batch2), (metrics, k)


@pytest.fixture()
def imperfect_case() -> Any:
    batch1 = {
        "embeddings": torch.stack([oh(0), oh(1), oh(3)]),  # 3d embedding pretends to be an error
        "labels": torch.tensor([0, 1, 1]),
        "is_query": torch.tensor([True, True, True]),
        "is_gallery": torch.tensor([False, False, False]),
        "categories": torch.tensor([10, 20, 20]),
    }

    batch2 = {
        "embeddings": torch.stack([oh(0), oh(1), oh(1)]),
        "labels": torch.tensor([0, 1, 1]),
        "is_query": torch.tensor([False, False, False]),
        "is_gallery": torch.tensor([True, True, True]),
        "categories": torch.tensor([10, 20, 20]),
    }

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics["OVERALL"]["cmc"][k] = 0.6666666865348816  # it's 2/3 in float precision
    metrics[10]["cmc"][k] = 1.0
    metrics[20]["cmc"][k] = 0.5

    return (batch1, batch2), (metrics, k)


@pytest.fixture()
def worst_case() -> Any:
    batch1 = {
        "embeddings": torch.stack([oh(1), oh(0), oh(0)]),  # 3d embedding pretends to be an error
        "labels": torch.tensor([0, 1, 1]),
        "is_query": torch.tensor([True, True, True]),
        "is_gallery": torch.tensor([False, False, False]),
        "categories": torch.tensor([10, 20, 20]),
    }

    batch2 = {
        "embeddings": torch.stack([oh(0), oh(1), oh(1)]),
        "labels": torch.tensor([0, 1, 1]),
        "is_query": torch.tensor([False, False, False]),
        "is_gallery": torch.tensor([True, True, True]),
        "categories": torch.tensor([10, 20, 20]),
    }

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics["OVERALL"]["cmc"][k] = 0
    metrics[10]["cmc"][k] = 0
    metrics[20]["cmc"][k] = 0

    return (batch1, batch2), (metrics, k)


def run_retrieval_metrics(case) -> None:  # type: ignore
    (batch1, batch2), (gt_metrics, k) = case

    top_k = (k,)

    num_samples = len(batch1["labels"]) + len(batch2["labels"])
    calc = EmbeddingMetrics(
        embeddings_key="embeddings",
        labels_key="labels",
        is_query_key="is_query",
        is_gallery_key="is_gallery",
        categories_key="categories",
        cmc_top_k=top_k,
        precision_top_k=top_k,
        map_top_k=top_k,
    )

    calc.setup(num_samples=num_samples)
    calc.update_data(batch1)
    calc.update_data(batch2)

    metrics = calc.compute_metrics()

    assert gt_metrics == metrics

    # the euclidean distance between any one-hots is always sqrt(2) or 0
    assert torch.isclose(calc.distance_matrix.unique(), torch.tensor([0, math.sqrt(2)])).all()  # type: ignore

    assert (calc.mask_gt.unique() == torch.tensor([0, 1])).all()  # type: ignore
    assert calc.acc.collected_samples == num_samples  # type: ignore


def run_across_epochs(case1, case2) -> None:  # type: ignore
    (batch11, batch12), (gt_metrics1, k1) = case1
    (batch21, batch22), (gt_metrics2, k2) = case2
    assert k1 == k2

    top_k = (k1,)

    calc = EmbeddingMetrics(
        embeddings_key="embeddings",
        labels_key="labels",
        is_query_key="is_query",
        is_gallery_key="is_gallery",
        categories_key="categories",
        cmc_top_k=top_k,
        precision_top_k=top_k,
        map_top_k=top_k,
    )

    def epoch_case(batch_a, batch_b, ground_truth_metrics) -> None:  # type: ignore
        num_samples = len(batch_a["labels"]) + len(batch_b["labels"])
        calc.setup(num_samples=num_samples)
        calc.update_data(batch_a)
        calc.update_data(batch_b)
        metrics = calc.compute_metrics()
        assert metrics == ground_truth_metrics

        # the euclidean distance between any one-hots is always sqrt(2) or 0
        assert torch.isclose(calc.distance_matrix.unique(), torch.tensor([0, math.sqrt(2)])).all()  # type: ignore

        assert (calc.mask_gt.unique() == torch.tensor([0, 1])).all()  # type: ignore
        assert calc.acc.collected_samples == num_samples

    # 1st epoch
    epoch_case(batch11, batch12, gt_metrics1)

    # 2nd epoch
    epoch_case(batch21, batch22, gt_metrics2)

    # 3d epoch
    epoch_case(batch11, batch12, gt_metrics1)

    # 4th epoch
    epoch_case(batch21, batch22, gt_metrics2)


def test_perfect_case(perfect_case) -> None:  # type: ignore
    run_retrieval_metrics(perfect_case)


def test_imperfect_case(imperfect_case) -> None:  # type: ignore
    run_retrieval_metrics(imperfect_case)


def test_worst_case(worst_case) -> None:  # type: ignore
    run_retrieval_metrics(worst_case)


def test_mixed_epochs(perfect_case, imperfect_case, worst_case):  # type: ignore
    cases = [perfect_case, imperfect_case, worst_case]
    for case1 in cases:
        for case2 in cases:
            run_across_epochs(case1, case2)
