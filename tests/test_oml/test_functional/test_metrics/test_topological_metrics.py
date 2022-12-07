from typing import Callable, Dict, Tuple

import pytest
import torch

from oml.functional.metrics import PCA, TMetricsDict, calc_pcf, calc_topological_metrics
from oml.utils.misc import remove_unused_kargs


def compare_dicts_recursively(d1: Dict, d2: Dict) -> bool:  # type: ignore
    assert set(d1.keys()) == set(
        d2.keys()
    ), f"The dictionaries keys are different.\nDict_1 keys: {set(d1.keys())}\nDict_2 keys: {set(d2.keys())}"
    for k, v in d1.items():
        if isinstance(v, dict):
            assert compare_dicts_recursively(
                v, d2[k]
            ), f"The dictionaries differs at key {k}.\nDict_1 value: {v}\nDict_2 value: {d2[k]}"
        else:
            assert d2[k] == v, f"Key name: {k}\nDict_1 value: {v}\nDict_2 value: {d2[k]}"
    return True


@pytest.fixture()
def eye_case() -> Tuple[torch.Tensor, TMetricsDict]:
    embeddings = torch.eye(300, 10, dtype=torch.float)
    embeddings = torch.cat((embeddings, embeddings), dim=1)

    metrics_expected: TMetricsDict = dict()
    metrics_expected["pcf"] = {0.5: torch.tensor(0.25), 0.9: torch.tensor(0.45)}
    return embeddings, metrics_expected


def test_calc_topological_metrics(eye_case: Tuple[torch.Tensor, TMetricsDict]) -> None:
    embeddings, metrics_expected = eye_case
    args = {"explained_variance_ths": tuple(metrics_expected["pcf"].keys())}
    metrics_evaluated = calc_topological_metrics(embeddings, **args)
    compare_dicts_recursively(metrics_evaluated, metrics_expected)


def test_pca() -> None:
    embeddings = 1.0 / (2 + torch.arange(10 * 6, dtype=torch.float).view(10, 6))
    sklearn_singular_values = torch.tensor(
        [6.1971074e-01, 5.2592568e-02, 1.9191128e-03, 3.9749077e-05, 4.8444451e-07, 2.5064631e-08]
    )
    sklearn_explained_variance = torch.tensor(
        [4.2671267e-02, 3.0733092e-04, 4.0922154e-07, 1.7555435e-10, 2.6076275e-14, 6.9803972e-17]
    )
    sklearn_explained_variance_ratio = torch.tensor(
        [9.9283969e-01, 7.1507213e-03, 9.5214282e-06, 4.0846531e-09, 6.0672116e-13, 1.6241410e-15]
    )
    sklearn_components = torch.tensor(
        [
            [0.7127961, 0.46693176, 0.34374583, 0.2697738, 0.22048034, 0.18532604],
            [0.5826674, -0.04804552, -0.2940672, -0.40069896, -0.4458503, -0.46083915],
            [0.35050747, -0.534863, -0.41878924, -0.08423905, 0.26763785, 0.58047855],
            [-0.16200304, 0.58922964, -0.20090438, -0.5153108, -0.2192341, 0.52210355],
            [0.05620524, -0.36120754, 0.6229757, -0.06948248, -0.5943722, 0.34664848],
            [-0.01268156, 0.12619649, -0.43762717, 0.69942355, -0.52880436, 0.15352166],
        ]
    )
    sklearn_mean = torch.tensor([0.09030403, 0.07110852, 0.06062918, 0.0537749, 0.04881614, 0.04499362])
    sklearn_embeddings_transformed = torch.tensor(
        [
            [5.63134372e-01, 1.41947456e-02, 7.17746589e-05, -1.11253485e-07, -2.88850703e-08, 2.56189630e-08],
            [8.04887488e-02, -3.82668823e-02, -1.07845606e-03, 7.57914177e-06, -1.80289916e-08, -4.66755878e-09],
            [-1.06218969e-02, -1.98824741e-02, 7.88418751e-04, -2.65598119e-05, 1.77929948e-07, -8.09011347e-09],
            [-5.01922816e-02, -8.27821996e-03, 8.46980955e-04, 3.52431857e-06, -3.00473857e-07, -8.16552159e-09],
            [-7.24049732e-02, -7.40899413e-04, 5.75136684e-04, 1.47551300e-05, -6.43953371e-08, -1.20967210e-08],
            [-8.66451785e-02, 4.49034991e-03, 2.63222348e-04, 1.43729976e-05, 1.36800239e-07, -9.23718080e-09],
            [-9.65574235e-02, 8.31926242e-03, -2.40651425e-05, 8.70206441e-06, 1.90188644e-07, -8.10177347e-09],
            [-1.03856489e-01, 1.12383962e-02, -2.74912774e-04, 9.26697680e-07, 1.29684949e-07, -1.11437739e-08],
            [-1.09456219e-01, 1.35357389e-02, -4.91006242e-04, -7.45797479e-06, -1.20018964e-08, -9.13348419e-09],
            [-1.13888584e-01, 1.53900003e-02, -6.77114527e-04, -1.57555241e-05, -1.92807832e-07, -1.15324976e-08],
        ]
    )
    pca = PCA(embeddings)
    embeddings_ = pca.transform(embeddings)

    assert torch.all(torch.isclose(sklearn_singular_values, pca.singular_values))
    assert torch.all(torch.isclose(sklearn_explained_variance, pca.explained_variance))
    assert torch.all(torch.isclose(sklearn_explained_variance_ratio, pca.explained_variance_ratio))
    assert torch.all(torch.isclose(sklearn_mean, pca.mean))
    assert torch.all(torch.isclose(sklearn_components, pca.components, atol=1e-3))
    assert torch.all(torch.isclose(sklearn_embeddings_transformed, embeddings_, atol=1.0e-4))


def test_pca_inverse_transform() -> None:
    embeddings = torch.eye(7, 6, dtype=torch.float)
    embeddings = torch.cat((embeddings, embeddings), dim=1)
    pca = PCA(embeddings)
    embeddings_t = pca.transform(embeddings)
    embeddings_it = pca.inverse_transform(embeddings_t)
    assert torch.all(torch.isclose(embeddings, embeddings_it, atol=1.0e-6))


def test_components_orthogonality() -> None:
    embeddings = torch.eye(300, 2, dtype=torch.float)
    embeddings = torch.cat((embeddings, embeddings), dim=1)
    pca = PCA(embeddings)
    assert torch.all(
        torch.isclose(torch.matmul(pca.components, pca.components.T), torch.eye(embeddings.shape[1]), atol=1.0e-6)
    )


@pytest.mark.parametrize(["metric_name", "metric_func"], [("pcf", calc_pcf)])
def test_calc_functions(
    eye_case: Tuple[torch.Tensor, TMetricsDict],
    metric_name: str,
    metric_func: Callable[[torch.Tensor, Tuple[int, ...]], torch.Tensor],
) -> None:
    embeddings, metrics_expected = eye_case
    explained_variance_to_keep = tuple(metrics_expected[metric_name].keys())
    kwargs = {"embeddings": embeddings, "explained_variance_ths": explained_variance_to_keep}

    kwargs = remove_unused_kargs(kwargs, metric_func)
    main_components_percentage = metric_func(**kwargs)  # type: ignore
    metrics_calculated = dict(zip(explained_variance_to_keep, main_components_percentage))
    for p in metrics_expected[metric_name].keys():
        values_expected = metrics_expected[metric_name][p]
        values_calculated = metrics_calculated[p]
        assert torch.all(
            torch.isclose(values_expected, values_calculated, atol=1e-4)
        ), f"Metric name: {metric_name}\nParameter value: {p}\nMetric function args: {kwargs}"
