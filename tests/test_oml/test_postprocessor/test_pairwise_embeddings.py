# type: ignore
# todo 522: remove ignore after we've done
import math
from functools import partial
from random import randint, random
from typing import Tuple

import pytest
import torch
from torch import BoolTensor, FloatTensor, Tensor

from oml.datasets.base import EmbeddingsQueryGalleryDataset
from oml.functional.metrics import calc_distance_matrix, calc_retrieval_metrics
from oml.interfaces.models import IPairwiseModel
from oml.models.meta.siamese import LinearTrivialDistanceSiamese
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.misc import flatten_dict, one_hot
from oml.utils.misc_torch import normalise, pairwise_dist

FEAT_SIZE = 8
oh = partial(one_hot, dim=FEAT_SIZE)


@pytest.fixture
def independent_query_gallery_case() -> Tuple[FloatTensor, BoolTensor, BoolTensor]:
    sz = 7
    feat_dim = 12

    embeddings = torch.randn((sz, feat_dim))
    embeddings = normalise(embeddings).float()

    is_query = torch.ones(sz).bool()
    is_query[: sz // 2] = False

    is_gallery = torch.ones(sz).bool()
    is_gallery[sz // 2 :] = False

    return embeddings, is_query, is_gallery


@pytest.fixture
def shared_query_gallery_case() -> Tuple[FloatTensor, BoolTensor, BoolTensor]:
    sz = 7
    feat_dim = 4

    embeddings = torch.randn((sz, feat_dim))
    embeddings = normalise(embeddings).float()

    is_query = torch.ones(sz).bool()
    is_gallery = torch.ones(sz).bool()

    return embeddings, is_query, is_gallery


@pytest.mark.long
@pytest.mark.parametrize("top_n, k", [(5, 5), (3, 4), (4, 3), (100, 5)])
@pytest.mark.parametrize("fixture_name", ["independent_query_gallery_case", "shared_query_gallery_case"])
def test_trivial_processing_does_not_change_distances_order(
    request: pytest.FixtureRequest, fixture_name: str, top_n: int, k: int
) -> None:
    embeddings, is_query, is_gallery = request.getfixturevalue(fixture_name)

    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=embeddings, is_query=is_query, is_gallery=is_gallery, labels=torch.ones_like(is_gallery).long()
    )

    distances = calc_distance_matrix(embeddings, is_query, is_gallery)

    # todo 522: refactor
    distances, retrieved_ids = torch.topk(distances, k=min(k, distances.shape[1]), largest=False)

    model = LinearTrivialDistanceSiamese(feat_dim=embeddings.shape[-1], identity_init=True)
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, num_workers=0, batch_size=64)

    distances_processed, retrieved_ids_upd = processor.process(
        distances=distances, retrieved_ids=retrieved_ids, dataset=dataset
    )

    assert (retrieved_ids == retrieved_ids_upd).all()
    assert torch.isclose(distances, distances_processed).all()


def perfect_case() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    query_labels = torch.tensor([1, 2, 3]).long()
    query_embeddings = torch.stack([oh(1), oh(2), oh(3)])

    gallery_labels = torch.tensor([1, 2, 1, 2, 3]).long()
    gallery_embeddings = torch.stack([oh(1), oh(2), oh(1), oh(2), oh(3)])

    return query_embeddings, gallery_embeddings, query_labels, gallery_labels


@pytest.mark.skip(reason="todo 522: rework later")
@pytest.mark.long
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
        model = LinearTrivialDistanceSiamese(feat_dim=gallery_embeddings.shape[-1], identity_init=True)
        processor = PairwiseReranker(pairwise_model=model, top_n=top_n, batch_size=16, num_workers=0)
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

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.distances_to_return


@pytest.mark.skip(reason="todo 522: rework later")
@pytest.mark.long
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
    processor = PairwiseReranker(pairwise_model=model, top_n=2, batch_size=128, num_workers=0)
    distances_upd = processor.process(
        distances=distances, queries=torch.randn((1, FEAT_SIZE)), galleries=torch.randn((5, FEAT_SIZE))
    )
    precisions_upd = calc_retrieval_metrics(distances=distances_upd, **args)["precision"]
    assert math.isclose(precisions_upd[1], 1)
    assert math.isclose(precisions_upd[3], 2 / 3, abs_tol=1e-5)


class RandomPairwise(IPairwiseModel):
    def __init__(self):  # type: ignore
        super(RandomPairwise, self).__init__()
        self.parameter = torch.nn.Linear(1, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return torch.sigmoid(torch.rand(x1.shape[0]))

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1, x2)


@pytest.mark.skip(reason="todo 522: rework test ")
@pytest.mark.long
@pytest.mark.parametrize("top_n", [2, 4, 5])
def test_processing_not_changing_non_sensitive_metrics(top_n: int) -> None:
    # The idea of the test is that postprocessing of first n elements
    # cannot change cmc@n and precision@n

    # Let's construct some random input
    query_embeddings_perfect, gallery_embeddings_perfect, query_labels, gallery_labels = perfect_case()
    query_embeddings = torch.rand_like(query_embeddings_perfect)
    gallery_embeddings = torch.rand_like(gallery_embeddings_perfect)
    mask_gt = query_labels.unsqueeze(-1) == gallery_labels

    distances = pairwise_dist(query_embeddings, gallery_embeddings)

    args = {
        "cmc_top_k": (top_n,),
        "precision_top_k": (top_n,),
        "fmr_vals": tuple(),
        "map_top_k": tuple(),
        "mask_gt": mask_gt,
    }

    metrics_before = calc_retrieval_metrics(distances=distances, **args)

    model = RandomPairwise()
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, batch_size=4, num_workers=0)
    distances_upd = processor.process(distances=distances, queries=query_embeddings, galleries=gallery_embeddings)

    metrics_after = calc_retrieval_metrics(distances=distances_upd, **args)

    assert metrics_before == metrics_after
