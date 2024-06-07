from typing import Callable, Tuple

import pytest
import torch

from oml.functional.metrics import TMetricsDict, calc_pcf, calc_topological_metrics
from oml.utils.misc import compare_dicts_recursively, remove_unused_kwargs


@pytest.fixture()
def eye_case() -> Tuple[torch.Tensor, TMetricsDict]:
    embeddings = torch.eye(300, 10, dtype=torch.float)
    embeddings = torch.cat((embeddings, embeddings), dim=1)

    metrics_expected: TMetricsDict = dict()
    metrics_expected["pcf"] = {0.5: torch.tensor(0.25), 0.9: torch.tensor(0.45)}
    return embeddings, metrics_expected


def test_calc_topological_metrics(eye_case: Tuple[torch.Tensor, TMetricsDict]) -> None:
    embeddings, metrics_expected = eye_case
    args = {"pcf_variance": tuple(metrics_expected["pcf"].keys()), "verbose": False}
    metrics_evaluated = calc_topological_metrics(embeddings, **args)  # type: ignore
    compare_dicts_recursively(metrics_evaluated, metrics_expected)


@pytest.mark.parametrize(["metric_name", "metric_func"], [("pcf", calc_pcf)])
def test_calc_functions(
    eye_case: Tuple[torch.Tensor, TMetricsDict],
    metric_name: str,
    metric_func: Callable[[torch.Tensor, Tuple[int, ...]], torch.Tensor],
) -> None:
    embeddings, metrics_expected = eye_case
    pcf_variance = tuple(metrics_expected[metric_name].keys())
    kwargs = {"embeddings": embeddings, "pcf_variance": pcf_variance}

    kwargs = remove_unused_kwargs(kwargs, metric_func)
    main_components_percentage = metric_func(**kwargs)  # type: ignore
    metrics_calculated = dict(zip(pcf_variance, main_components_percentage))
    for p in metrics_expected[metric_name].keys():
        values_expected = metrics_expected[metric_name][p]
        values_calculated = metrics_calculated[p]
        assert torch.all(
            torch.isclose(values_expected, values_calculated, atol=1e-4)
        ), f"Metric name: {metric_name}\nParameter value: {p}\nMetric function args: {kwargs}"
