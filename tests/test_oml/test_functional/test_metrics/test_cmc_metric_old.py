from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch import Tensor

from oml.functional.metrics import calc_retrieval_metrics
from oml.utils.misc_torch import pairwise_dist


def cmc_score_count(distances: Tensor, mask_gt: Tensor, topk: int, mask_to_ignore: Optional[Tensor] = None) -> float:
    metrics = calc_retrieval_metrics(
        distances=distances,
        mask_gt=mask_gt,
        mask_to_ignore=mask_to_ignore,
        cmc_top_k=(topk,),
        map_top_k=tuple(),
        precision_top_k=tuple(),
        fmr_vals=tuple(),
    )
    return metrics["cmc"][topk]


def cmc_score(
    query_embeddings: Tensor,
    gallery_embeddings: Tensor,
    mask_gt: Tensor,
    topk: int,
    mask_to_ignore: Optional[Tensor] = None,
) -> float:
    distances = pairwise_dist(x1=query_embeddings, x2=gallery_embeddings, p=2)
    val = cmc_score_count(distances=distances, mask_gt=mask_gt, topk=topk, mask_to_ignore=mask_to_ignore)
    return val


EPS = 1e-4

TEST_DATA_SIMPLE = (
    # (distance_matrix, mask_gt,  topk, expected_value)
    (torch.tensor([[1, 2], [2, 1]]), torch.tensor([[0, 1], [1, 0]]), 1, 0.0),
    (torch.tensor([[0, 0.5], [0.0, 0.5]]), torch.tensor([[0, 1], [1, 0]]), 1, 0.5),
    (torch.tensor([[0, 0.5], [0.0, 0.5]]), torch.tensor([[0, 1], [1, 0]]), 2, 1),
    (
        torch.tensor([[1, 0.5, 0.2], [2, 3, 4], [0.4, 3, 4]]),
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        2,
        1 / 3,
    ),
    (torch.randn((10, 10)), torch.ones((10, 10)), 1, 1),
)

TEST_DATA_LESS_SMALL = (
    (torch.rand((10, 10)) + torch.tril(torch.ones((10, 10))), torch.eye(10), i, i / 10) for i in range(1, 10)
)

TEST_DATA_GREATER_SMALL = (
    (
        torch.rand((10, 10)) + torch.triu(torch.ones((10, 10)), diagonal=1),
        torch.eye(10),
        i,
        i / 10,
    )
    for i in range(1, 10)
)

TEST_DATA_LESS_BIG = (
    (
        torch.rand((100, 100)) + torch.tril(torch.ones((100, 100))),
        torch.eye(100),
        i,
        i / 100,
    )
    for i in range(1, 101, 10)
)


@pytest.mark.parametrize("distance_matrix,mask_gt,topk,expected", TEST_DATA_SIMPLE)
def test_metric_count(distance_matrix: Tensor, mask_gt: Tensor, topk: int, expected: float) -> None:
    """Simple test"""
    out = cmc_score_count(distances=distance_matrix, mask_gt=mask_gt, topk=topk)
    assert np.isclose(out, expected), (out, expected, "xxx")


@pytest.mark.parametrize(
    "distance_matrix,mask_gt,topk,expected",
    chain(TEST_DATA_LESS_SMALL, TEST_DATA_LESS_BIG),
)
def test_metric_less(distance_matrix: Tensor, mask_gt: Tensor, topk: int, expected: float) -> None:
    """Simple test"""
    out = cmc_score_count(distances=distance_matrix, mask_gt=mask_gt, topk=topk)
    assert out - EPS <= expected


@pytest.mark.parametrize("distance_matrix,mask_gt,topk,expected", chain(TEST_DATA_GREATER_SMALL))
def test_metric_greater(distance_matrix: Tensor, mask_gt: Tensor, topk: int, expected: float) -> None:
    """Simple test"""
    out = cmc_score_count(distances=distance_matrix, mask_gt=mask_gt, topk=topk)
    assert out + EPS >= expected


@pytest.fixture
def generate_samples_for_cmc_score() -> List[Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate list of query and gallery data for cmc score testing.
    """
    data = []
    for error_rate in [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
    ]:
        # generate params of the datasets
        class_number = np.random.randint(low=2, high=5)
        kq = np.random.randint(low=500, high=1000)
        kg = np.random.randint(low=250, high=500)

        def generate_samples(n_labels, samples_per_label):  # type: ignore
            samples = []  # type: ignore
            labels = []  # type: ignore
            # for each label generate dots that will be close to each other and
            # distanced from samples of other labels
            for i in range(n_labels):
                tmp_samples = np.random.uniform(low=2 * i, high=2 * i + 0.2, size=(samples_per_label,))
                samples = np.concatenate((samples, tmp_samples))
                labels = np.concatenate((labels, [i] * samples_per_label))
            return samples.reshape((-1, 1)), labels  # type: ignore

        query_embs, query_labels = generate_samples(n_labels=class_number, samples_per_label=kq)

        gallery_embs, gallery_labels = generate_samples(n_labels=class_number, samples_per_label=kg)

        # spoil generated gallery dataset: for each sample from data change
        # label to any other one with probability error_rate
        def confuse_labels(labels, error_rate):  # type: ignore
            unique_labels = set(labels)
            size = len(labels)
            for i in range(size):
                if np.random.binomial(n=1, p=error_rate, size=1)[0]:
                    labels[i] = np.random.choice(list(unique_labels - {labels[i]}))
            return labels

        gallery_labels = confuse_labels(gallery_labels, error_rate=error_rate)

        query_embs = torch.tensor(query_embs)
        gallery_embs = torch.tensor(gallery_embs)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        gallery_labels = torch.tensor(gallery_labels, dtype=torch.long)

        data.append((error_rate, query_embs, query_labels, gallery_embs, gallery_labels))
    return data


def test_cmc_score_with_samples(generate_samples_for_cmc_score) -> None:  # type: ignore
    """
    Count cmc score callback for sets of well-separated data clusters labeled
    with error_rate probability mistake.
    """
    for (
        error_rate,
        query_embs,
        query_labels,
        gallery_embs,
        gallery_labels,
    ) in generate_samples_for_cmc_score:
        true_cmc_01 = 1 - error_rate
        mask_gt = (query_labels.reshape((-1, 1)) == gallery_labels).to(torch.bool)
        cmc = cmc_score(
            query_embeddings=query_embs,
            gallery_embeddings=gallery_embs,
            mask_gt=mask_gt,
            topk=1,
        )
        assert abs(cmc - true_cmc_01) <= 0.05


@pytest.mark.parametrize(
    (
        "query_embeddings",
        "gallery_embeddings",
        "mask_gt",
        "available_samples",
        "topk",
        "expected",
    ),
    (
        (
            torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 1], [0, 0, 1, 1]]).float(),
            torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]]).float(),
            torch.tensor(
                [
                    [True, False, False],
                    [True, False, False],
                    [False, True, True],
                    [False, True, True],
                ]
            ),
            torch.tensor(
                [
                    [False, True, True],
                    [True, True, True],
                    [True, False, True],
                    [True, True, True],
                ]
            ),
            1,
            0.75,
        ),
        (
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]).float(),
            torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 1]]).float(),
            torch.tensor(
                [
                    [False, False, True],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ]
            ),
            torch.tensor(
                [
                    [True, True, True],
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ]
            ),
            1,
            0.25,
        ),
    ),
)
def test_masked_cmc_score(
    query_embeddings: Tensor,
    gallery_embeddings: Tensor,
    mask_gt: Tensor,
    available_samples: Tensor,
    topk: int,
    expected: float,
) -> None:
    score = cmc_score(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        mask_gt=mask_gt,
        mask_to_ignore=~available_samples,
        topk=topk,
    )
    assert score == expected


@pytest.mark.parametrize(
    (
        "query_embeddings",
        "gallery_embeddings",
        "mask_gt",
        "available_samples",
        "topk",
    ),
    (
        (
            torch.rand(size=(query_size, 32)).float(),
            torch.rand(size=(gallery_size, 32)).float(),
            torch.randint(low=0, high=2, size=(query_size, gallery_size)).bool(),
            torch.ones(size=(query_size, gallery_size)).bool(),
            k,
        )
        for query_size, gallery_size, k in zip(list(range(10, 20)), list(range(25, 35)), list(range(1, 11)))
    ),
)
def test_no_mask_cmc_score(
    query_embeddings: Tensor, gallery_embeddings: Tensor, mask_gt: Tensor, available_samples: Tensor, topk: int
) -> None:
    """
    In this test we just check that masked_cmc_score is equal to cmc_score
    when all the samples are available for for scoring.
    """
    masked_score = cmc_score(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        mask_gt=mask_gt,
        mask_to_ignore=~available_samples,
        topk=topk,
    )
    score = cmc_score(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        mask_gt=mask_gt,
        mask_to_ignore=None,
        topk=topk,
    )
    assert masked_score == score
