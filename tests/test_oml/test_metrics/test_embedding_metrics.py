import math
from collections import defaultdict
from functools import partial
from typing import Any, Tuple

import pytest
import torch

from oml.const import (
    CATEGORIES_KEY,
    EMBEDDINGS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    OVERALL_CATEGORIES_KEY,
    PATHS_KEY,
)
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import one_hot

oh = partial(one_hot, dim=8)


def check_dicts_of_dicts_are_equal(dict1: Any, dict2: Any) -> None:
    for key1 in dict1:
        for key2 in dict1[key1]:
            assert dict1[key1][key2] == dict2[key1][key2]

    for key1 in dict2:
        for key2 in dict2[key1]:
            assert dict2[key1][key2] == dict1[key1][key2]


@pytest.fixture()
def perfect_case() -> Any:
    """
    Here we assume that our model provides the best possible embeddings:
    for the item with the label parameter equals to i,
    it provides one hot vector with non zero element in i-th position.

    Thus, we expect all of the metrics equals to 1.
    """

    batch1 = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(1), oh(1)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([True, True, True]),
        IS_GALLERY_KEY: torch.tensor([False, False, False]),
        CATEGORIES_KEY: ["cat", "dog", "dog"],
    }

    batch2 = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(1), oh(1)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([False, False, False]),
        IS_GALLERY_KEY: torch.tensor([True, True, True]),
        CATEGORIES_KEY: ["cat", "dog", "dog"],
    }

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 1.0
    metrics["cat"]["cmc"][k] = 1.0
    metrics["dog"]["cmc"][k] = 1.0

    return (batch1, batch2), (metrics, k)


@pytest.fixture()
def imperfect_case() -> Any:
    batch1 = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(1), oh(3)]),  # 3d embedding pretends to be an error
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([True, True, True]),
        IS_GALLERY_KEY: torch.tensor([False, False, False]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
    }

    batch2 = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(1), oh(1)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([False, False, False]),
        IS_GALLERY_KEY: torch.tensor([True, True, True]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
    }

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 0.6666666865348816  # it's 2/3 in float precision
    metrics[10]["cmc"][k] = 1.0
    metrics[20]["cmc"][k] = 0.5

    return (batch1, batch2), (metrics, k)


@pytest.fixture()
def worst_case() -> Any:
    batch1 = {
        EMBEDDINGS_KEY: torch.stack([oh(1), oh(0), oh(0)]),  # 3d embedding pretends to be an error
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([True, True, True]),
        IS_GALLERY_KEY: torch.tensor([False, False, False]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
    }

    batch2 = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(1), oh(1)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([False, False, False]),
        IS_GALLERY_KEY: torch.tensor([True, True, True]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
    }

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 0
    metrics[10]["cmc"][k] = 0
    metrics[20]["cmc"][k] = 0

    return (batch1, batch2), (metrics, k)


@pytest.fixture()
def case_for_distance_check() -> Any:
    batch1 = {
        EMBEDDINGS_KEY: torch.stack([oh(1) * 2, oh(1) * 3, oh(0)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([True, True, True]),
        IS_GALLERY_KEY: torch.tensor([False, False, False]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
    }

    batch2 = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(1), oh(1)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([False, False, False]),
        IS_GALLERY_KEY: torch.tensor([True, True, True]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
    }
    ids_ranked_by_distance = [0, 2, 1]
    return (batch1, batch2), ids_ranked_by_distance


def run_retrieval_metrics(case) -> None:  # type: ignore
    (batch1, batch2), (gt_metrics, k) = case

    top_k = (k,)

    num_samples = len(batch1[LABELS_KEY]) + len(batch2[LABELS_KEY])
    calc = EmbeddingMetrics(
        embeddings_key=EMBEDDINGS_KEY,
        labels_key=LABELS_KEY,
        is_query_key=IS_QUERY_KEY,
        is_gallery_key=IS_GALLERY_KEY,
        categories_key=CATEGORIES_KEY,
        cmc_top_k=top_k,
        precision_top_k=tuple(),
        map_top_k=tuple(),
        fmr_vals=tuple(),
    )

    calc.setup(num_samples=num_samples)
    calc.update_data(batch1)
    calc.update_data(batch2)

    metrics = calc.compute_metrics()

    check_dicts_of_dicts_are_equal(gt_metrics, metrics)

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
        embeddings_key=EMBEDDINGS_KEY,
        labels_key=LABELS_KEY,
        is_query_key=IS_QUERY_KEY,
        is_gallery_key=IS_GALLERY_KEY,
        categories_key=CATEGORIES_KEY,
        cmc_top_k=top_k,
        precision_top_k=tuple(),
        map_top_k=tuple(),
        fmr_vals=tuple(),
    )

    def epoch_case(batch_a, batch_b, ground_truth_metrics) -> None:  # type: ignore
        num_samples = len(batch_a[LABELS_KEY]) + len(batch_b[LABELS_KEY])
        calc.setup(num_samples=num_samples)
        calc.update_data(batch_a)
        calc.update_data(batch_b)
        metrics = calc.compute_metrics()

        check_dicts_of_dicts_are_equal(metrics, ground_truth_metrics)

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


def test_worst_k(case_for_distance_check) -> None:  # type: ignore
    (batch1, batch2), gt_ids = case_for_distance_check

    num_samples = len(batch1[LABELS_KEY]) + len(batch2[LABELS_KEY])
    calc = EmbeddingMetrics(
        embeddings_key=EMBEDDINGS_KEY,
        labels_key=LABELS_KEY,
        is_query_key=IS_QUERY_KEY,
        is_gallery_key=IS_GALLERY_KEY,
        categories_key=CATEGORIES_KEY,
        cmc_top_k=(),
        precision_top_k=(),
        map_top_k=(2,),
        fmr_vals=tuple(),
    )

    calc.setup(num_samples=num_samples)
    calc.update_data(batch1)
    calc.update_data(batch2)

    calc.compute_metrics()

    assert calc.get_worst_queries_ids(f"{OVERALL_CATEGORIES_KEY}/map/2", 3) == gt_ids


@pytest.mark.parametrize("extra_keys", [[], [PATHS_KEY], [PATHS_KEY, "a"], ["a"]])
def test_ready_to_vis(extra_keys: Tuple[str, ...]) -> None:  # type: ignore
    calc = EmbeddingMetrics(
        embeddings_key=EMBEDDINGS_KEY,
        labels_key=LABELS_KEY,
        is_query_key=IS_QUERY_KEY,
        is_gallery_key=IS_GALLERY_KEY,
        categories_key=CATEGORIES_KEY,
        extra_keys=extra_keys,
        cmc_top_k=(1,),
        precision_top_k=(),
        map_top_k=(),
        fmr_vals=tuple(),
    )

    assert calc.ready_to_visualize() or PATHS_KEY not in extra_keys
