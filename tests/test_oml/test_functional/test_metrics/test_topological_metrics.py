from typing import Callable, Tuple

import pytest
import torch

from oml.functional.metrics import (
    TMetricsDict,
    calc_main_components_percentage,
    calc_topological_metrics,
)
from oml.utils.misc import remove_unused_kargs


def compare_dicts(d1: dict, d2: dict) -> None:  # type: ignore
    for k, v in d1.items():
        assert d2[k] == v, [k, v]


def compare_dicts_reciprocally(d1: dict, d2: dict) -> None:  # type: ignore
    compare_dicts(d1, d2)
    compare_dicts(d2, d1)


@pytest.fixture()
def test_case() -> Tuple[torch.Tensor, TMetricsDict]:
    embeddings = torch.eye(300, 10, dtype=torch.float)
    embeddings = torch.cat((embeddings, embeddings), dim=1)

    metrics_expected: TMetricsDict = dict()
    metrics_expected["main_components"] = {50: torch.tensor(25.0), 90: torch.tensor(45.0)}
    return embeddings, metrics_expected


def test_calc_topological_metrics(test_case: Tuple[torch.Tensor, TMetricsDict]) -> None:
    embeddings, metrics_expected = test_case
    args = {"explained_variance_to_keep": tuple(metrics_expected["main_components"].keys())}
    metrics_evaluated = calc_topological_metrics(embeddings, **args)
    compare_dicts_reciprocally(metrics_evaluated, metrics_expected)


@pytest.mark.parametrize(["metric_name", "metric_func"], [("main_components", calc_main_components_percentage)])
def test_calc_functions(
    test_case: Tuple[torch.Tensor, TMetricsDict],
    metric_name: str,
    metric_func: Callable[[torch.Tensor, Tuple[int, ...]], torch.Tensor],
) -> None:
    embeddings, metrics_expected = test_case
    explained_variance_to_keep = tuple(metrics_expected[metric_name].keys())
    kwargs = {"embeddings": embeddings, "explained_variance_to_keep": explained_variance_to_keep}

    kwargs = remove_unused_kargs(kwargs, metric_func)
    main_components_percentage = metric_func(**kwargs)  # type: ignore
    metrics_calculated = dict(zip(explained_variance_to_keep, main_components_percentage))
    for p in metrics_expected[metric_name].keys():
        values_expected = metrics_expected[metric_name][p]
        values_calculated = metrics_calculated[p]
        assert torch.all(
            torch.isclose(values_expected, values_calculated, atol=1e-4)
        ), f"Metric name: {metric_name}\nParameter value: {p}\nMetric function args: {kwargs}"
