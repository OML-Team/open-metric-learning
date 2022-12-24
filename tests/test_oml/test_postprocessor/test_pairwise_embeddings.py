from functools import partial
from random import randint, random
from typing import Tuple

import pytest
import torch
from torch import Tensor

from oml.functional.metrics import calc_distance_matrix, calc_retrieval_metrics
from oml.models.siamese import SimpleSiamese
from oml.postprocessors.pairwise_embeddings import PairwiseEmbeddingsPostprocessor
from oml.utils.misc import flatten_dict, one_hot
from oml.utils.misc_torch import pairwise_dist

oh = partial(one_hot, dim=8)


@pytest.fixture
def independent_query_gallery_case() -> Tuple[Tensor, Tensor, Tensor]:
    sz = 7
    feat_dim = 12

    embeddings = torch.randn((sz, feat_dim))

    is_query = torch.ones(sz).bool()
    is_query[: sz // 2] = False

    is_gallery = torch.ones(sz).bool()
    is_gallery[sz // 2 :] = False

    return embeddings, is_query, is_gallery


@pytest.fixture
def shared_query_gallery_case() -> Tuple[Tensor, Tensor, Tensor]:
    sz = 7
    feat_dim = 4

    embeddings = torch.randn((sz, feat_dim))
    is_query = torch.ones(sz).bool()
    is_gallery = torch.ones(sz).bool()

    return embeddings, is_query, is_gallery


@pytest.mark.parametrize("top_n", [1, 2, 5, 100])
@pytest.mark.parametrize("fixture_name", ["independent_query_gallery_case", "shared_query_gallery_case"])
def test_identity_processing(request: pytest.FixtureRequest, fixture_name: str, top_n: int) -> None:
    embeddings, is_query, is_gallery = request.getfixturevalue(fixture_name)
    embeddings_query = embeddings[is_query]
    embeddings_gallery = embeddings[is_gallery]

    distances = calc_distance_matrix(embeddings, is_query, is_gallery)

    id_model = SimpleSiamese(feat_dim=embeddings.shape[-1], identity_init=True)
    id_processor = PairwiseEmbeddingsPostprocessor(pairwise_model=id_model, top_n=top_n)

    distances_processed = id_processor.process(
        emb_query=embeddings_query,
        emb_gallery=embeddings_gallery,
        distances=distances.clone(),
    )

    order = distances.argsort()
    order_processed = distances_processed.argsort()

    assert (order == order_processed).all(), (order, order_processed)


def perfect_case() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    query_labels = torch.tensor([1, 2, 3]).long()
    query_embeddings = torch.stack([oh(1), oh(2), oh(3)])

    gallery_labels = torch.tensor([1, 2, 1, 2, 3]).long()
    gallery_embeddings = torch.stack([oh(1), oh(2), oh(1), oh(2), oh(3)])

    return query_embeddings, gallery_embeddings, query_labels, gallery_labels


def test_processing_fixes_broken_perfect_case() -> None:
    """
    The idea of the test is the following:

    1. We generate perfect set of labels and one-hot embeddings representing them
    2. We make distances matrix broken by randomly replacing some distance with a random value
    3. We apply pairwise postprocessor which simply restores l2 distances in the distances matrix
    4. No matter what other parameters were, metrics must be the same as before or become better

    """
    n_repetitions = 20
    for _ in range(n_repetitions):

        query_embeddings, gallery_embeddings, query_labels, gallery_labels = perfect_case()
        distances = pairwise_dist(query_embeddings, gallery_embeddings)
        mask_gt = query_labels.unsqueeze(-1) == gallery_labels

        nq, ng = distances.shape

        # Let's randomly change some distances to break the case
        for _ in range(5):
            i = randint(0, nq - 1)
            j = randint(0, ng - 1)
            distances[i, j] = random()

        # As mentioned before, for this test the exact values of parameters don't matter
        top_k = (randint(1, ng - 1),)
        top_n = randint(1, 10)

        args = {"mask_gt": mask_gt, "precision_top_k": top_k, "map_top_k": top_k, "cmc_top_k": top_k, "fmr_vals": ()}

        # Metrics before
        metrics = flatten_dict(calc_retrieval_metrics(distances=distances, **args))

        # Metrics after broken distances have been fixed
        id_model = SimpleSiamese(feat_dim=gallery_embeddings.shape[-1], identity_init=True)
        processor = PairwiseEmbeddingsPostprocessor(pairwise_model=id_model, top_n=top_n)
        distances_upd = processor.process(distances, query_embeddings, gallery_embeddings)
        metrics_upd = flatten_dict(calc_retrieval_metrics(distances=distances_upd, **args))

        for key in metrics.keys():
            metric = metrics[key]
            metric_upd = metrics_upd[key]
            assert metric_upd >= metric, (key, metric, metric_upd)
