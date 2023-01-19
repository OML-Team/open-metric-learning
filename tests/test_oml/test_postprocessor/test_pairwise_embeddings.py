import math
from functools import partial
from random import randint, random
from typing import Tuple

import pytest
import torch
from torch import Tensor

from oml.functional.metrics import calc_distance_matrix, calc_retrieval_metrics
from oml.interfaces.models import IPairwiseModel
from oml.models.siamese import LinearSiamese
from oml.retrieval.postprocessors.pairwise import PairwiseEmbeddingsPostprocessor
from oml.utils.misc import flatten_dict, one_hot
from oml.utils.misc_torch import normalise, pairwise_dist

FEAT_SIZE = 8
oh = partial(one_hot, dim=FEAT_SIZE)


@pytest.fixture
def independent_query_gallery_case() -> Tuple[Tensor, Tensor, Tensor]:
    sz = 7
    feat_dim = 12

    embeddings = torch.randn((sz, feat_dim))
    embeddings = normalise(embeddings)

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
    embeddings = normalise(embeddings)

    is_query = torch.ones(sz).bool()
    is_gallery = torch.ones(sz).bool()

    return embeddings, is_query, is_gallery


@pytest.mark.parametrize("top_n", [2, 5, 100])
@pytest.mark.parametrize("fixture_name", ["independent_query_gallery_case", "shared_query_gallery_case"])
def test_trivial_processing_does_not_change_distances_order(
    request: pytest.FixtureRequest, fixture_name: str, top_n: int
) -> None:
    embeddings, is_query, is_gallery = request.getfixturevalue(fixture_name)
    embeddings_query = embeddings[is_query]
    embeddings_gallery = embeddings[is_gallery]

    distances = calc_distance_matrix(embeddings, is_query, is_gallery)

    model = LinearSiamese(feat_dim=embeddings.shape[-1], identity_init=True)
    processor = PairwiseEmbeddingsPostprocessor(pairwise_model=model, top_n=top_n, num_workers=0, batch_size=64)

    distances_processed = processor.process(
        queries=embeddings_query,
        galleries=embeddings_gallery,
        distances=distances.clone(),
    )

    order = distances.argsort()
    order_processed = distances_processed.argsort()

    assert (order == order_processed).all(), (order, order_processed)

    if top_n <= is_gallery.sum():
        min_orig_distances = torch.topk(distances, k=top_n, largest=False).values
        min_processed_distances = torch.topk(distances_processed, k=top_n, largest=False).values
        assert torch.allclose(min_orig_distances, min_processed_distances)


def perfect_case() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    query_labels = torch.tensor([1, 2, 3]).long()
    query_embeddings = torch.stack([oh(1), oh(2), oh(3)])

    gallery_labels = torch.tensor([1, 2, 1, 2, 3]).long()
    gallery_embeddings = torch.stack([oh(1), oh(2), oh(1), oh(2), oh(3)])

    return query_embeddings, gallery_embeddings, query_labels, gallery_labels


def test_trivial_processing_fixes_broken_perfect_case() -> None:
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
        top_n = randint(2, 10)

        args = {"mask_gt": mask_gt, "precision_top_k": top_k, "map_top_k": top_k, "cmc_top_k": top_k, "fmr_vals": ()}

        # Metrics before
        metrics = flatten_dict(calc_retrieval_metrics(distances=distances, **args))

        # Metrics after broken distances have been fixed
        model = LinearSiamese(feat_dim=gallery_embeddings.shape[-1], identity_init=True)
        processor = PairwiseEmbeddingsPostprocessor(pairwise_model=model, top_n=top_n, batch_size=16, num_workers=0)
        distances_upd = processor.process(distances, query_embeddings, gallery_embeddings)
        metrics_upd = flatten_dict(calc_retrieval_metrics(distances=distances_upd, **args))

        for key in metrics.keys():
            metric = metrics[key]
            metric_upd = metrics_upd[key]
            assert metric_upd >= metric, (key, metric, metric_upd)


class DummyPairwise(IPairwiseModel):
    def __init__(self, distances_to_return: Tensor):
        super(DummyPairwise, self).__init__()
        self.distances_to_return = distances_to_return
        self.parameter = torch.nn.Linear(1, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.distances_to_return


def test_trivial_processing_fixes_broken_perfect_case_2() -> None:
    """
    The idea of the test is similar to "test_trivial_processing_fixes_broken_perfect_case",
    but this time we check the exact metrics values.

    """
    distances = torch.tensor([[0.8, 0.3, 0.2, 0.4, 0.5]])
    mask_gt = torch.tensor([[1, 1, 0, 1, 0]]).bool()

    args = {"mask_gt": mask_gt, "precision_top_k": (1, 3)}

    precisions = calc_retrieval_metrics(distances=distances, **args)["precision"]
    assert math.isclose(precisions[1], 0)
    assert math.isclose(precisions[3], 2 / 3, abs_tol=1e-5)

    # Now let's fix the error with dummy pairwise model
    model = DummyPairwise(distances_to_return=torch.tensor([3.5, 2.5]))
    processor = PairwiseEmbeddingsPostprocessor(pairwise_model=model, top_n=2, batch_size=128, num_workers=0)
    distances_upd = processor.process(
        distances=distances, queries=torch.randn((1, FEAT_SIZE)), galleries=torch.randn((5, FEAT_SIZE))
    )
    precisions_upd = calc_retrieval_metrics(distances=distances_upd, **args)["precision"]
    assert math.isclose(precisions_upd[1], 1)
    assert math.isclose(precisions_upd[3], 2 / 3, abs_tol=1e-5)
