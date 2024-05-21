import math
from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from oml.const import CATEGORIES_COLUMN, OVERALL_CATEGORIES_KEY
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models.meta.siamese import LinearTrivialDistanceSiamese
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.misc import compare_dicts_recursively, flatten_dict, one_hot
from tests.test_integrations.utils import EmbeddingsQueryGalleryLabeledDataset

FEAT_DIM = 8
oh = partial(one_hot, dim=FEAT_DIM)


def get_trivial_postprocessor(top_n: int) -> PairwiseReranker:
    model = LinearTrivialDistanceSiamese(feat_dim=FEAT_DIM, identity_init=True)
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, num_workers=0, batch_size=64)
    return processor


@pytest.fixture()
def perfect_case() -> Any:
    """
    Here we assume that our model provides the best possible embeddings:
    for the item with the label parameter equals to i,
    it provides one hot vector with non zero element in i-th position.

    Thus, we expect all of the metrics equals to 1.
    """
    dataset = EmbeddingsQueryGalleryLabeledDataset(
        embeddings=torch.stack([oh(0), oh(1), oh(1), oh(0), oh(1), oh(1)]).float(),
        labels=torch.tensor([0, 1, 1, 0, 1, 1]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=np.array(["cat", "dog", "dog", "cat", "dog", "dog"]),
    )

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 1.0
    metrics["cat"]["cmc"][k] = 1.0
    metrics["dog"]["cmc"][k] = 1.0

    return dataset, (metrics, k)


@pytest.fixture()
def imperfect_case() -> Any:
    dataset = EmbeddingsQueryGalleryLabeledDataset(
        embeddings=torch.stack([oh(0), oh(1), oh(3), oh(0), oh(1), oh(1)]).float(),  # 3d val pretends to be an error
        labels=torch.tensor([0, 1, 1, 0, 1, 1]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=np.array([10, 20, 20, 10, 20, 20]),
    )

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 0.6666666865348816  # it's 2/3 in float precision
    metrics[10]["cmc"][k] = 1.0
    metrics[20]["cmc"][k] = 0.5

    return dataset, (metrics, k)


@pytest.fixture()
def worst_case() -> Any:
    dataset = EmbeddingsQueryGalleryLabeledDataset(
        embeddings=torch.stack([oh(1), oh(0), oh(0), oh(0), oh(1), oh(1)]).float(),  # all are errors
        labels=torch.tensor([0, 1, 1, 0, 1, 1]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=np.array([10, 20, 20, 10, 20, 20]),
    )

    k = 1
    metrics = defaultdict(lambda: defaultdict(dict))  # type: ignore
    metrics[OVERALL_CATEGORIES_KEY]["cmc"][k] = 0
    metrics[10]["cmc"][k] = 0
    metrics[20]["cmc"][k] = 0

    return dataset, (metrics, k)


@pytest.fixture()
def case_for_finding_worst_queries() -> Any:
    dataset = EmbeddingsQueryGalleryLabeledDataset(
        embeddings=torch.stack([oh(0), oh(1), oh(2), oh(0), oh(5), oh(5)]).float(),  # last 2 are errors
        labels=torch.tensor([0, 1, 2, 0, 1, 2]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
        categories=np.array([10, 20, 20, 10, 20, 20]),
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
        postprocessor=get_trivial_postprocessor(top_n=num_samples),
    )

    calc.setup()

    for batch in DataLoader(dataset, batch_size=4, shuffle=False):
        calc.update(embeddings=batch[dataset.input_tensors_key], indices=batch[dataset.index_key])

    metrics = calc.compute_metrics()

    compare_dicts_recursively(gt_metrics, metrics)

    # the euclidean distance between any one-hots is always sqrt(2) or 0
    for distances in calc.retrieval_results.distances:  # type: ignore
        assert (
            torch.isclose(distances, torch.tensor([0.0])).any()
            or torch.isclose(distances, torch.tensor([math.sqrt(2)])).any()
        )

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
        postprocessor=get_trivial_postprocessor(top_n=num_samples),
    )

    metrics_all_epochs = []

    for _ in range(2):  # epochs
        calc.setup()

        for batch in DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False, drop_last=False):
            calc.update(embeddings=batch[dataset.input_tensors_key], indices=batch[dataset.index_key])

        metrics_all_epochs.append(calc.compute_metrics())

    assert compare_dicts_recursively(metrics_all_epochs[0], metrics_all_epochs[-1])

    # the euclidean distance between any one-hots is always sqrt(2) or 0
    for distances in calc.retrieval_results.distances:  # type: ignore
        assert (
            torch.isclose(distances, torch.tensor([0.0])).any()
            or torch.isclose(distances, torch.tensor([math.sqrt(2)])).any()
        )

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
    dataset, worst_query_ids = case_for_finding_worst_queries

    calc = EmbeddingMetrics(
        dataset=dataset,
        cmc_top_k=(1,),
        precision_top_k=(),
        map_top_k=(),
        fmr_vals=tuple(),
        postprocessor=get_trivial_postprocessor(top_n=len(dataset)),
    )

    calc.setup()
    for batch in DataLoader(dataset, batch_size=4, shuffle=False):
        calc.update(embeddings=batch[dataset.input_tensors_key], indices=batch[dataset.index_key])

    calc.compute_metrics()

    assert set(calc.get_worst_queries_ids(f"{OVERALL_CATEGORIES_KEY}/cmc/1", 2)) == worst_query_ids


def test_all_requested_metrics_are_calculated(perfect_case) -> None:  # type: ignore
    dataset, _ = perfect_case

    calc = EmbeddingMetrics(
        dataset=dataset,
        cmc_top_k=(1,),
        precision_top_k=(2,),
        map_top_k=(4, 500),
        pcf_variance=(0.2, 0.1),
        fmr_vals=(0.3, 0.5),
        postprocessor=get_trivial_postprocessor(top_n=len(dataset)),
    )

    calc.setup()
    for batch in DataLoader(dataset, batch_size=4, shuffle=False):
        calc.update(embeddings=batch[dataset.input_tensors_key], indices=batch[dataset.index_key])

    metrics = calc.compute_metrics()
    metrics = flatten_dict(metrics)

    for category_key in [OVERALL_CATEGORIES_KEY, *np.unique(dataset.extra_data[CATEGORIES_COLUMN])]:
        assert metrics.pop(f"{category_key}/cmc/1") == 1
        assert metrics.pop(f"{category_key}/precision/2") == 1
        assert metrics.pop(f"{category_key}/map/4") == 1
        assert metrics.pop(f"{category_key}/map/500") == 1
        assert metrics.pop(f"{category_key}/pcf/0.2") is not None
        assert metrics.pop(f"{category_key}/pcf/0.1") is not None

    assert metrics.pop(f"{OVERALL_CATEGORIES_KEY}/fnmr@fmr/0.3") == 0
    assert metrics.pop(f"{OVERALL_CATEGORIES_KEY}/fnmr@fmr/0.5") == 0

    assert not metrics, "There are unwilling extra keys."
