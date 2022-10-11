from typing import Any, Dict

import numpy as np
import pytest
import torch

from oml.functional.metrics import reduce_metrics


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
