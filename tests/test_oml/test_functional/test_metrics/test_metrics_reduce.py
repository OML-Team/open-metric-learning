from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch import BoolTensor

from oml.functional.metrics import reduce_metrics, take_unreduced_metrics_by_mask
from oml.utils.misc import compare_dicts_recursively


@pytest.mark.parametrize(
    "metrics_dict,result_dict",
    [
        ({"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}}, {"OVERALL": {"cmc": {1: 0.1}}}),
        ({"cmc": {1: 1}}, {"cmc": {1: 1}}),
        ({"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}, "a": 0.3}, {"OVERALL": {"cmc": {1: 0.1}}, "a": 0.3}),
        (
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}, "a": torch.tensor([0.3, 0.1])},
            {"OVERALL": {"cmc": {1: 0.1}}, "a": 0.2},
        ),
        (
            {"OVERALL": {"cmc": {1: np.r_[0.1, 0.2, 0.0]}}, "a": np.r_[0.3, 0.1]},
            {"OVERALL": {"cmc": {1: np.mean(np.r_[0.1, 0.2, 0.0])}}, "a": np.mean(np.r_[0.3, 0.1])},
        ),
        (
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}, "a": torch.tensor([0.3])},
            {"OVERALL": {"cmc": {1: 0.1}}, "a": 0.3},
        ),
        pytest.param({"OVERALL": {"cmc": {1: {}}}}, {"OVERALL": {"cmc": {}}}, marks=pytest.mark.xfail),
    ],
)
def test_reduce_metrics(metrics_dict: Dict[str, Any], result_dict: Dict[str, Any]) -> None:
    assert reduce_metrics(metrics_dict) == result_dict, reduce_metrics(metrics_dict)


@pytest.mark.parametrize(
    "metrics_dict,mask,result_dict",
    [
        # nothing to mask
        ({"cmc": {1: 1}}, torch.tensor([0, 1, 1, 0]).bool(), {"cmc": {1: 1}}),
        # mask 1 element
        (
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}},
            torch.tensor([0, 1, 0]).bool(),
            {"OVERALL": {"cmc": {1: torch.tensor([0.2])}}},
        ),
        # mask 0 elements
        (
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}},
            torch.tensor([0, 0, 0]).bool(),
            {"OVERALL": {"cmc": {1: torch.tensor([])}}},
        ),
        # mask all
        (
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}},
            torch.tensor([1, 1, 1]).bool(),
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.2, 0.0])}}},
        ),
        # mask some
        (
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.5, 0.0])}}},
            torch.tensor([1, 1, 0]).bool(),
            {"OVERALL": {"cmc": {1: torch.tensor([0.1, 0.5])}}},
        ),
        # mask some + scalars
        (
            {"OVERALL": {"a": 1, "cmc": {1: torch.tensor([0.1, 0.5, 0.0])}}},
            torch.tensor([1, 1, 0]).bool(),
            {"OVERALL": {"a": 1, "cmc": {1: torch.tensor([0.1, 0.5])}}},
        ),
        # empty
        ({}, torch.tensor([1, 1, 0]).bool(), {}),
    ],
)
def test_take_unreduced_metrics_by_mask(
    metrics_dict: Dict[str, Any], mask: BoolTensor, result_dict: Dict[str, Any]
) -> None:
    output = take_unreduced_metrics_by_mask(metrics_dict, mask)
    assert compare_dicts_recursively(output, result_dict)
