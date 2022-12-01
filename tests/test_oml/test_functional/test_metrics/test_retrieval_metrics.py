from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pytest
import torch

from oml.functional.losses import surrogate_precision
from oml.functional.metrics import (
    TMetricsDict,
    calc_fnmr_at_fmr,
    calc_gt_mask,
    calc_mask_to_ignore,
    calc_retrieval_metrics,
    validate_dataset,
)

from .synthetic import generate_distance_matrix, generate_retrieval_case

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
        num_gt = min(len(pos), k)
        values[query_idx] = sum(gt_count / (idx + 1) for gt_count, idx in enumerate(pos, 1) if idx < k) / num_gt
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


def test_on_exact_case() -> None:
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

    metrics_expected = defaultdict(dict)  # type: ignore

    metrics_expected["precision"][1] = torch.tensor([1, 1, 0, 0, 1, 0]).float()
    metrics_expected["map"][1] = torch.tensor([1, 1, 0, 0, 1, 0]).float()

    metrics_expected["precision"][2] = torch.tensor([1, 0.5, 0.5, 0, 1, 0]).float()
    metrics_expected["map"][2] = torch.tensor([1, 0.5, 0.25, 0, 1, 0]).float()

    metrics_expected["precision"][3] = torch.tensor([0.6666, 0.3333, 0.6666, 0.3333, 1, 0]).float()
    metrics_expected["map"][3] = torch.tensor([0.6666, 0.3333, 0.3888, 0.1111, 1, 0]).float()

    metrics_expected["precision"][4] = torch.tensor([1, 0.6666, 0.6666, 0.6666, 1, 1]).float()
    metrics_expected["map"][4] = torch.tensor([0.9166, 0.5, 0.3888, 0.2777, 1, 0.25]).float()

    metrics_expected["precision"][5] = torch.tensor([1, 1, 1, 1, 1, 1]).float()
    metrics_expected["map"][5] = torch.tensor([0.9166, 0.7, 0.5888, 0.4777, 1, 0.25]).float()

    metrics_expected["precision"][10] = metrics_expected["precision"][5]
    metrics_expected["map"][10] = metrics_expected["map"][5]

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


def test_validate_dataset_good_case() -> None:
    isq = np.r_[True, False, False, True, False, False]
    isg = np.r_[False, True, True, False, True, True]
    labels = np.r_[0, 0, 0, 1, 1, 1]

    mgt = calc_gt_mask(labels=labels, is_query=isq, is_gallery=isg)
    m2i = calc_mask_to_ignore(is_query=isq, is_gallery=isg)
    validate_dataset(mask_gt=mgt, mask_to_ignore=m2i)


def test_validate_dataset_bad_case() -> None:
    with pytest.raises(AssertionError):
        isq = np.r_[True, False, False, True, True]
        isg = np.r_[False, True, True, False, True]
        labels = np.r_[0, 0, 0, 1, 1]

        mgt = calc_gt_mask(labels=labels, is_query=isq, is_gallery=isg)
        m2i = calc_mask_to_ignore(is_query=isq, is_gallery=isg)
        validate_dataset(mask_gt=mgt, mask_to_ignore=m2i)


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


def test_calc_fnmr_at_fmr() -> None:
    fmr_vals = (10, 50)
    pos_dist = torch.tensor([0, 0, 1, 1, 2, 2, 5, 5, 9, 9])
    neg_dist = torch.tensor([3, 3, 4, 4, 6, 6, 7, 7, 8, 8])
    fnmr_at_fmr = calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals)
    # 10 percentile of negative distances is 3 and
    # the number of positive distances that are greater than
    # or equal to 3 is 4 so FNMR@FMR(10%) is 4 / 10
    # 50 percentile of negative distances is 6 and
    # the number of positive distances that are greater than
    # or equal to 6 is 2 so FNMR@FMR(50%) is 2 / 10
    assert torch.all(torch.isclose(fnmr_at_fmr, torch.tensor([0.4, 0.2])))
