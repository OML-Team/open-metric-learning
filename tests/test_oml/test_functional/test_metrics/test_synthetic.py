from typing import List, Tuple

import pytest
import torch

from oml.functional.metrics import (
    apply_mask_to_ignore,
    calc_gt_mask,
    calc_mask_to_ignore,
)

from .synthetic import generate_distance_matrix, generate_retrieval_case

TCase = Tuple[List[List[int]], torch.Tensor, torch.Tensor, torch.Tensor]


def case_1() -> TCase:
    labels = torch.tensor([2, 2, 2, 0, 0, 0, 0, 1, 1])
    is_query = torch.tensor([True, False, True, False, True, False, True, False, True])
    is_gallery = torch.tensor([True] * len(labels))

    desired_positions = [[1, 3], [2, 4], [0, 3, 5], [5, 6, 7], [1]]

    return desired_positions, labels, is_query, is_gallery


def case_2() -> TCase:
    labels = torch.tensor([2, 2, 2, 0, 0, 0, 0, 1, 1])
    is_query = torch.tensor([True, False, True, False, True, False, True, False, True])
    is_gallery = torch.logical_not(is_query)

    desired_positions = [[2], [0], [0, 1], [2, 3], [3]]

    return desired_positions, labels, is_query, is_gallery


@pytest.mark.parametrize("case", [case_1(), case_2()])
def test_exact_cases(case: TCase) -> None:
    check_distance_matrix(case)


@pytest.mark.parametrize("max_num_labels", list(range(3, 10)))
@pytest.mark.parametrize("max_num_samples_per_label", list(range(3, 10)))
@pytest.mark.parametrize("is_query_all", [True, False])
@pytest.mark.parametrize("is_gallery_all", [True, False])
@pytest.mark.parametrize("num_attempts", [20])
def test_on_synthetic_cases(
    max_num_labels: int, max_num_samples_per_label: int, num_attempts: int, is_query_all: bool, is_gallery_all: bool
) -> None:
    for _ in range(num_attempts):
        case = generate_retrieval_case(
            max_labels=max_num_labels,
            max_samples_per_label=max_num_samples_per_label,
            is_query_all=is_query_all,
            is_gallery_all=is_gallery_all,
            return_desired_correct_positions=True,
        )
        check_distance_matrix(case)


def check_distance_matrix(case: TCase) -> None:
    desired_positions_array, labels, is_query, is_gallery = case
    distances = generate_distance_matrix(
        desired_positions_array, labels=labels, is_query=is_query, is_gallery=is_gallery
    )

    mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)
    mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)
    distances, mask_gt = apply_mask_to_ignore(distances=distances, mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)

    _, ids_sorted_distance = torch.topk(distances, largest=False, k=distances.shape[1])

    for query_label, ids_sorted_for_query, desired_positions in zip(
        labels[is_query], ids_sorted_distance, desired_positions_array
    ):
        actual_positions = torch.nonzero(labels[is_gallery][ids_sorted_for_query] == query_label)
        actual_positions = actual_positions[: len(desired_positions)].squeeze(1).tolist()
        assert actual_positions == desired_positions, [actual_positions, desired_positions]
