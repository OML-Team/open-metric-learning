from typing import Any

import pytest

from oml.utils.misc import flatten_dict


@pytest.fixture()
def case1() -> Any:
    input_dict = {1: {"dog": 0.5, "cat": 0.3}, 5: {"dog": 0.7, "cat": 0.2}, 7: {10: {"dog": 0.5, "cat": 0.3}}}

    target_dict = {
        "metric/1/dog": 0.5,
        "metric/1/cat": 0.3,
        "metric/5/dog": 0.7,
        "metric/5/cat": 0.2,
        "metric/7/10/dog": 0.5,
        "metric/7/10/cat": 0.3,
    }

    return input_dict, target_dict


def test_converter(case1) -> None:  # type: ignore
    input_dict, target_dict_gt = case1

    target_dict = flatten_dict(input_dict, parent_key="metric")

    assert target_dict_gt == target_dict
