import math
from collections import defaultdict
from functools import partial
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from oml.const import OVERALL_CATEGORIES_KEY
from oml.datasets.base import EmbeddingsQueryGalleryDataset
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models.meta.siamese import LinearTrivialDistanceSiamese
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.misc import compare_dicts_recursively, one_hot

FEAT_DIM = 8
oh = partial(one_hot, dim=FEAT_DIM)


def get_trivial_postprocessor(top_n: int) -> PairwiseReranker:
    model = LinearTrivialDistanceSiamese(feat_dim=FEAT_DIM, identity_init=True)
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, num_workers=0, batch_size=64)
    return processor


def compare_tensors_as_sets(x: Tensor, y: Tensor, decimal_tol: int = 4) -> bool:
    set_x = torch.round(x, decimals=decimal_tol).unique()
    set_y = torch.round(y, decimals=decimal_tol).unique()
    return bool(torch.isclose(set_x, set_y).all())


@pytest.fixture()
def perfect_case() -> Any:
    """
    Here we assume that our model provides the best possible embeddings:
    for the item with the label parameter equals to i,
    it provides one hot vector with non zero element in i-th position.

    Thus, we expect all of the metrics equals to 1.
    """
    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=torch.stack([oh(0), oh(1), oh(1), oh(0), oh(1), oh(1)]).float(),
        labels=torch.tensor([0, 1, 1, 0, 1, 1]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=["cat", "dog", "dog", "cat", "dog", "dog"],
    )

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 1.0
    metrics["cat"]["cmc"][k] = 1.0
    metrics["dog"]["cmc"][k] = 1.0

    return dataset, (metrics, k)


@pytest.fixture()
def imperfect_case() -> Any:
    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=torch.stack([oh(0), oh(1), oh(3), oh(0), oh(1), oh(1)]).float(),  # 3d val pretends to be an error
        labels=torch.tensor([0, 1, 1, 0, 1, 1]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=torch.tensor([10, 20, 20, 10, 20, 20]).long(),
    )

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 0.6666666865348816  # it's 2/3 in float precision
    metrics[10]["cmc"][k] = 1.0
    metrics[20]["cmc"][k] = 0.5

    return dataset, (metrics, k)


@pytest.fixture()
def worst_case() -> Any:
    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=torch.stack([oh(1), oh(0), oh(0), oh(0), oh(1), oh(1)]).float(),  # all are errors
        labels=torch.tensor([0, 1, 1, 0, 1, 1]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=torch.tensor([10, 20, 20, 10, 20, 20]).long(),
    )

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 0
    metrics[10]["cmc"][k] = 0
    metrics[20]["cmc"][k] = 0

    return dataset, (metrics, k)


@pytest.fixture()
def case_for_finding_worst_queries() -> Any:
    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=torch.stack([oh(0), oh(1), oh(2), oh(0), oh(5), oh(5)]).float(),  # last 2 are errors
        labels=torch.tensor([0, 1, 2, 0, 1, 2]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=torch.tensor([10, 20, 20, 10, 20, 20]).long(),
    )

    worst_two_queries = {1, 2}
    return dataset, worst_two_queries


def run_retrieval_metrics(case) -> None:  # type: ignore
    dataset, (gt_metrics, k) = case

    top_k = (k,)

    num_samples = len(dataset)
    calc = EmbeddingMetrics(
        dataset=dataset,
        cmc_top_k=top_k,
        precision_top_k=tuple(),
        map_top_k=tuple(),
        fmr_vals=tuple(),
        pcf_variance=tuple(),
        postprocessor=get_trivial_postprocessor(top_n=2),
    )

    calc.setup(num_samples=num_samples)

    for batch in DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False, drop_last=False):
        embeddings = batch[dataset.input_tensors_key]
        calc.update_data(embeddings=embeddings)

    metrics = calc.compute_metrics()

    assert compare_dicts_recursively(gt_metrics, metrics)

    # the euclidean distance between any one-hots is always sqrt(2) or 0
    assert compare_tensors_as_sets(calc.prediction.distances, torch.tensor([0, math.sqrt(2)]))

    assert calc.acc.collected_samples == num_samples


def run_across_epochs(case) -> None:  # type: ignore
    dataset, (gt_metrics, k) = case

    top_k = (k,)

    num_samples = len(dataset)
    calc = EmbeddingMetrics(
        dataset=dataset,
        cmc_top_k=top_k,
        precision_top_k=tuple(),
        map_top_k=tuple(),
        fmr_vals=tuple(),
        pcf_variance=tuple(),
        postprocessor=get_trivial_postprocessor(top_n=3),
    )

    metrics_all_epochs = []

    for _ in range(2):  # epochs
        calc.setup(num_samples=num_samples)

        for batch in DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False, drop_last=False):
            embeddings = batch[dataset.input_tensors_key]
            calc.update_data(embeddings=embeddings)

        metrics_all_epochs.append(calc.compute_metrics())

    assert compare_dicts_recursively(metrics_all_epochs[0], metrics_all_epochs[-1])

    # the euclidean distance between any one-hots is always sqrt(2) or 0
    assert compare_tensors_as_sets(calc.prediction.distances, torch.tensor([0, math.sqrt(2)]))

    assert calc.acc.collected_samples == num_samples


def test_perfect_case(perfect_case) -> None:  # type: ignore
    run_retrieval_metrics(perfect_case)


def test_imperfect_case(imperfect_case) -> None:  # type: ignore
    run_retrieval_metrics(imperfect_case)


def test_worst_case(worst_case) -> None:  # type: ignore
    run_retrieval_metrics(worst_case)


def test_several_epochs(perfect_case, imperfect_case, worst_case):  # type: ignore
    run_across_epochs(perfect_case)
    run_across_epochs(imperfect_case)
    run_across_epochs(worst_case)


def test_worst_k(case_for_finding_worst_queries) -> None:  # type: ignore
    dataset, worst_queries = case_for_finding_worst_queries

    num_samples = len(dataset)
    calc = EmbeddingMetrics(
        dataset=dataset,
        cmc_top_k=(1,),
        precision_top_k=(),
        map_top_k=(2,),
        fmr_vals=(0.2,),
        pcf_variance=(0.2,),
        postprocessor=get_trivial_postprocessor(top_n=1_000),
        verbose=False,
    )

    calc.setup(num_samples=num_samples)

    for batch in DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False, drop_last=False):
        embeddings = batch[dataset.input_tensors_key]
        calc.update_data(embeddings=embeddings)

    calc.compute_metrics()

    metric_name = f"{OVERALL_CATEGORIES_KEY}/cmc/1"
    assert set(calc.get_worst_queries_ids(metric_name, n_queries=len(worst_queries))) == worst_queries
