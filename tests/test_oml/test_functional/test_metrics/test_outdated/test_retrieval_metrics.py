from collections import defaultdict
from typing import Callable, List, Tuple

import numpy as np
import pytest
import torch
from torch import FloatTensor, LongTensor

from oml.functional.losses import surrogate_precision
from oml.functional.metrics import (
    TMetricsDict,
    apply_mask_to_ignore,
    calc_cmc,
    calc_fnmr_at_fmr,
    calc_gt_mask,
    calc_map,
    calc_mask_to_ignore,
    calc_precision,
)
from oml.functional.metrics import (
    calc_retrieval_metrics_on_full as calc_retrieval_metrics,
)
from oml.metrics import calc_fnmr_at_fmr_rr
from oml.retrieval import RetrievalResults
from oml.utils.misc import compare_dicts_recursively
from oml.utils.misc_torch import take_2d
from tests.test_oml.test_functional.test_metrics.test_outdated.synthetic import (
    generate_distance_matrix,
    generate_retrieval_case,
)

TPositions = List[List[int]]


def naive_cmc(positions: TPositions, k: int) -> torch.Tensor:
    values = torch.empty(len(positions), dtype=torch.bool)
    for query_idx, pos in enumerate(positions):
        values[query_idx] = any(idx < k for idx in pos)
    metric = torch.mean(values.float())
    return metric


def naive_map(positions: TPositions, k: int) -> torch.Tensor:
    values = torch.empty(len(positions), dtype=torch.float)
    for query_idx, pos in enumerate(positions):
        n_correct_before_j = {j: sum(el < j for el in pos[:j]) for j in range(1, k + 1)}
        values[query_idx] = sum(n_correct_before_j[i] / i * (i - 1 in pos) for i in range(1, k + 1)) / (
            n_correct_before_j[k] or float("inf")
        )
    metric = torch.mean(values.float())
    return metric


def naive_precision(positions: TPositions, k: int) -> torch.Tensor:
    values = torch.empty(len(positions), dtype=torch.float)
    for query_idx, pos in enumerate(positions):
        num_gt = min(len(pos), k)
        values[query_idx] = sum(1 for idx in pos if idx < k) / num_gt
    metric = torch.mean(values.float())
    return metric


def compare_with_approx_precision(
    positions: TPositions,
    labels: torch.Tensor,
    is_query: torch.Tensor,
    is_gallery: torch.Tensor,
    expected_metrics: TMetricsDict,
    top_k: Tuple[int, ...],
    reduction: str,
) -> None:
    expected_precision = expected_metrics["precision"]
    distances = generate_distance_matrix(positions, labels=labels, is_query=is_query, is_gallery=is_gallery)
    mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)

    mask_to_ignore = calc_mask_to_ignore(is_query, is_gallery)
    # we use this custom code instead of `apply_mask_to_ignore` to avoid having inf values
    distances[mask_to_ignore] = 1_000_000
    mask_gt[mask_to_ignore] = False

    for k in top_k:
        metrics_approx = surrogate_precision(distances, mask_gt, k, t1=1e-6, t2=1e-6, reduction=reduction)
        assert torch.isclose(metrics_approx, expected_precision[k], atol=1e-3).all()


TExactTestCase = Tuple[
    List[List[int]], torch.Tensor, torch.Tensor, torch.Tensor, TMetricsDict, Tuple[int, ...], torch.Tensor, torch.Tensor
]


@pytest.fixture()
def exact_test_case() -> TExactTestCase:
    """
    label 0:
    VVXVX
    VXXVV
    XVVXV
    XXVVV
    label 1:
    VXXXX
    XXXVX
    """
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    is_query = torch.tensor([True] * len(labels))
    is_gallery = torch.tensor([True] * len(labels))
    positions = [[0, 1, 3], [0, 3, 4], [1, 2, 4], [2, 3, 4], [0], [3]]
    top_k = (1, 2, 3, 4, 5, 10)
    max_k = min(len(labels), max(top_k))

    distances = generate_distance_matrix(positions, labels=labels, is_query=is_query, is_gallery=is_gallery)
    mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)
    mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)
    distances, mask_gt = apply_mask_to_ignore(distances=distances, mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)
    _, ii_top_k = torch.topk(distances, k=max_k, largest=False)
    gt_tops = take_2d(mask_gt, ii_top_k)

    metrics_expected = defaultdict(dict)  # type: ignore

    metrics_expected["cmc"][1] = torch.tensor([1, 1, 0, 0, 1, 0]).float()
    metrics_expected["precision"][1] = torch.tensor([1, 1, 0, 0, 1, 0]).float()
    metrics_expected["map"][1] = torch.tensor([1, 1, 0, 0, 1, 0]).float()

    metrics_expected["cmc"][2] = torch.tensor([1, 1, 1, 0, 1, 0]).float()
    metrics_expected["precision"][2] = torch.tensor([1, 0.5, 0.5, 0, 1, 0]).float()
    metrics_expected["map"][2] = torch.tensor([1, 1, 1 / 2, 0, 1, 0]).float()

    metrics_expected["cmc"][3] = torch.tensor([1, 1, 1, 1, 1, 0]).float()
    metrics_expected["precision"][3] = torch.tensor([0.6666, 0.3333, 0.6666, 0.3333, 1, 0]).float()
    metrics_expected["map"][3] = torch.tensor([1, 1, 7 / 12, 1 / 3, 1, 0]).float()

    metrics_expected["cmc"][4] = torch.tensor([1, 1, 1, 1, 1, 1]).float()
    metrics_expected["precision"][4] = torch.tensor([1, 0.6666, 0.6666, 0.6666, 1, 1]).float()
    metrics_expected["map"][4] = torch.tensor([11 / 12, 3 / 4, 7 / 12, 5 / 12, 1, 1 / 4]).float()

    metrics_expected["cmc"][5] = torch.tensor([1, 1, 1, 1, 1, 1]).float()
    metrics_expected["precision"][5] = torch.tensor([1, 1, 1, 1, 1, 1]).float()
    metrics_expected["map"][5] = torch.tensor([11 / 12, 21 / 30, 53 / 90, 43 / 90, 1, 1 / 4]).float()

    metrics_expected["cmc"][10] = torch.tensor([1, 1, 1, 1, 1, 1]).float()
    metrics_expected["precision"][10] = metrics_expected["precision"][5]
    metrics_expected["map"][10] = metrics_expected["map"][5]

    return positions, labels, is_query, is_gallery, metrics_expected, top_k, mask_gt, gt_tops


def test_on_exact_case(exact_test_case: TExactTestCase) -> None:
    positions, labels, is_query, is_gallery, metrics_expected, top_k, mask_gt, gt_tops = exact_test_case
    compare_metrics(positions, labels, is_query, is_gallery, metrics_expected, top_k, reduce=False)
    compare_with_approx_precision(positions, labels, is_query, is_gallery, metrics_expected, top_k, reduction="none")


@pytest.mark.parametrize("max_num_labels", list(range(5, 10)))
@pytest.mark.parametrize("max_num_samples_per_label", list(range(3, 10)))
@pytest.mark.parametrize("is_query_all", [True, False])
@pytest.mark.parametrize("is_gallery_all", [True, False])
@pytest.mark.parametrize("num_attempts", [20])
def test_on_synthetic_cases(
    max_num_labels: int, max_num_samples_per_label: int, num_attempts: int, is_query_all: bool, is_gallery_all: bool
) -> None:
    top_k = (1, 2, 3, 4, 5)

    for _ in range(num_attempts):
        case = generate_retrieval_case(
            max_labels=max_num_labels,
            max_samples_per_label=max_num_samples_per_label,
            is_query_all=is_query_all,
            is_gallery_all=is_gallery_all,
            return_desired_correct_positions=True,
        )
        positions, labels, is_query, is_gallery = case

        metrics_expected = defaultdict(dict)  # type: ignore

        metrics_expected["cmc"] = {k: naive_cmc(positions, k) for k in top_k}
        metrics_expected["precision"] = {k: naive_precision(positions, k) for k in top_k}
        metrics_expected["map"] = {k: naive_map(positions, k) for k in top_k}

        compare_metrics(positions, labels, is_query, is_gallery, metrics_expected, top_k, reduce=True)


def compare_metrics(
    positions: TPositions,
    labels: torch.Tensor,
    is_query: torch.Tensor,
    is_gallery: torch.Tensor,
    metrics_expected: TMetricsDict,
    top_k: Tuple[int, ...],
    reduce: bool,
) -> None:
    distances = generate_distance_matrix(positions, labels=labels, is_query=is_query, is_gallery=is_gallery)
    mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)
    mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)

    metrics_calculated = calc_retrieval_metrics(
        distances=distances,
        mask_gt=mask_gt,
        mask_to_ignore=mask_to_ignore,
        map_top_k=top_k,
        precision_top_k=top_k,
        cmc_top_k=top_k,
        reduce=reduce,
    )

    for metric_name in metrics_expected.keys():
        for k in top_k:
            values_expected = metrics_expected[metric_name][k]
            values_calculated = metrics_calculated[metric_name][k]
            assert torch.all(torch.isclose(values_expected, values_calculated, atol=1e-4)), [metric_name, k]


@pytest.mark.parametrize(
    "metric_function, metric_name", [(calc_cmc, "cmc"), (calc_precision, "precision"), (calc_map, "map")]
)
def test_metrics(
    metric_function: Callable[[torch.Tensor, torch.Tensor, Tuple[int, ...]], torch.Tensor],
    metric_name: str,
    exact_test_case: TExactTestCase,
) -> None:
    _, _, _, _, metrics_expected, top_k, mask_gt, gt_tops = exact_test_case
    kwargs = {"gt_tops": gt_tops, "n_gts": mask_gt.sum(dim=1), "top_k": top_k}
    metric_vals = metric_function(**kwargs)  # type: ignore
    for k, metric_val in zip(top_k, metric_vals):
        assert torch.all(
            torch.isclose(metric_val, metrics_expected[metric_name][k], atol=1.0e-4)
        ), f"{metric_name}@{k} expected: {metrics_expected[metric_name][k]}; evaluated: {metric_val}."


@pytest.mark.parametrize("top_k", (1, 2, 3, 4, 5, 10))
@pytest.mark.parametrize(
    "metric_function, metric_name", [(calc_cmc, "cmc"), (calc_precision, "precision"), (calc_map, "map")]
)
def test_metrics_individual(
    metric_function: Callable[[torch.Tensor, torch.Tensor, Tuple[int, ...]], torch.Tensor],
    metric_name: str,
    exact_test_case: TExactTestCase,
    top_k: int,
) -> None:
    _, _, _, _, metrics_expected, _, mask_gt, gt_tops = exact_test_case
    kwargs = {"gt_tops": gt_tops, "n_gts": mask_gt.sum(dim=1), "top_k": (top_k,)}
    metric_val = metric_function(**kwargs)[0]  # type: ignore
    assert torch.all(
        torch.isclose(metric_val, metrics_expected[metric_name][top_k], atol=1.0e-4)
    ), f"{metric_name}@{top_k} expected: {metrics_expected[metric_name][top_k]}; evaluated: {metric_val}."


@pytest.mark.parametrize("top_k", (tuple(), (0, -1), (0,), (1.5, 2), (1.0, 2.0)))
@pytest.mark.parametrize("metric_function", [calc_cmc, calc_precision, calc_map])
def test_metrics_check_params(
    metric_function: Callable[[torch.Tensor, torch.Tensor, Tuple[int, ...]], torch.Tensor], top_k: Tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        gt_tops = torch.ones((10, 5), dtype=torch.bool)
        n_gt = torch.ones(10)
        kwargs = {"gt_tops": gt_tops, "n_gts": n_gt, "top_k": (top_k,)}
        metric_function(**kwargs)  # type: ignore


def test_calc_fnmr_at_fmr() -> None:
    fmr_vals = (0.1, 0.5)
    pos_dist = np.array([0, 0, 1, 1, 2, 2, 5, 5, 9, 9], dtype=np.float32)
    neg_dist = np.array([3, 3, 4, 4, 6, 6, 7, 7, 8, 8], dtype=np.float32)
    fnmr_at_fmr = calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals)
    fnmr_at_fmr_expected = torch.tensor([0.4, 0.2])
    # 10 percentile of negative distances is 3 and
    # the number of positive distances that are greater than
    # or equal to 3 is 4 so FNMR@FMR(10%) is 40%
    # 50 percentile of negative distances is 6 and
    # the number of positive distances that are greater than
    # or equal to 6 is 2 so FNMR@FMR(50%) is 20%
    assert torch.all(
        torch.isclose(fnmr_at_fmr, fnmr_at_fmr_expected)
    ), f"fnmr@fmr({fmr_vals}),  expected: {fnmr_at_fmr_expected}; evaluated: {fnmr_at_fmr}."


def test_calc_fnmr_at_fmr_rr() -> None:
    # it's the same test data as above, but presented in different format
    dist = [
        FloatTensor([0, 0, 1, 1, 2, 2, 5, 5, 7, 7, 8, 8, 9]),
        FloatTensor([3, 3, 4, 4, 6, 6, 9]),
    ]

    # hint: correctly retrieved ids start with 1, others start with 2
    retrieved_ids = [
        LongTensor([10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 18]),
        LongTensor([20, 21, 22, 23, 24, 25, 10]),
    ]

    gt_ids = [
        LongTensor([10, 11, 12, 13, 14, 15, 16, 17, 18]),
        LongTensor([10]),
    ]

    rr = RetrievalResults(dist, retrieved_ids, gt_ids)

    fnmr_at_fmr = calc_fnmr_at_fmr_rr(rr, fmr_vals=(0.1, 0.5))
    fnmr_at_fmr_expected = {"fnmr@fmr": {0.1: torch.tensor(0.4), 0.5: torch.tensor(0.2)}}

    assert compare_dicts_recursively(fnmr_at_fmr, fnmr_at_fmr_expected)


@pytest.mark.parametrize("fmr_vals", (tuple(), (0, -1), (101,)))
def test_calc_fnmr_at_fmr_check_params(fmr_vals: Tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        pos_dist = np.zeros(10)
        neg_dist = np.ones(10)
        calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals)
