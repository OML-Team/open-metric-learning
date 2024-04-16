# type: ignore
from functools import partial
from random import randint, sample
from typing import Tuple

import pytest
import torch
from torch import BoolTensor, FloatTensor

from oml.datasets import EmbeddingsQueryGalleryDataset
from oml.functional.metrics import calc_retrieval_metrics
from oml.interfaces.models import IPairwiseModel
from oml.models.meta.siamese import LinearTrivialDistanceSiamese
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.retrieval.prediction import RetrievalPrediction
from oml.utils.misc import flatten_dict, one_hot
from oml.utils.misc_torch import normalise

FEAT_SIZE = 8
oh = partial(one_hot, dim=FEAT_SIZE)


@pytest.fixture
def independent_query_gallery_case() -> Tuple[FloatTensor, BoolTensor, BoolTensor]:
    sz = 7
    feat_dim = 12

    embeddings = torch.randn((sz, feat_dim)).float()
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

    embeddings = torch.randn((sz, feat_dim)).float()
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

    prediction = RetrievalPrediction.compute_from_embeddings(
        embeddings=embeddings, dataset=dataset, n_items_to_retrieve=k
    )

    model = LinearTrivialDistanceSiamese(feat_dim=embeddings.shape[-1], identity_init=True)
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, num_workers=0, batch_size=64)

    prediction_upd = processor.process(prediction, dataset=dataset)

    assert (prediction.retrieved_ids == prediction_upd.retrieved_ids).all()
    assert torch.isclose(prediction.distances, prediction_upd.distances, rtol=1e-6).all()


def perfect_case() -> EmbeddingsQueryGalleryDataset:
    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=torch.stack([oh(1), oh(2), oh(3), oh(1), oh(2), oh(1), oh(2), oh(3)]).float(),
        labels=torch.tensor([1, 2, 3, 1, 2, 1, 2, 3]).long(),
        is_query=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).bool(),
        is_gallery=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).bool(),
    )

    return dataset


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

        dataset = perfect_case()

        # Let's randomly swap some embeddings to break the case
        embeddings_broken = dataset._embeddings.clone()
        for _ in range(5):
            i, j = sample(list(range(len(dataset))), 2)
            embeddings_broken[i], embeddings_broken[j] = embeddings_broken[j], embeddings_broken[i]

        pred = RetrievalPrediction.compute_from_embeddings(embeddings=embeddings_broken, dataset=dataset)

        top_k = (randint(1, pred.distances.shape[1] - 1),)
        args = {"gt_ids": pred.gt_ids, "precision_top_k": top_k, "map_top_k": top_k, "cmc_top_k": top_k}

        # Metrics before
        metrics = flatten_dict(calc_retrieval_metrics(retrieved_ids=pred.retrieved_ids, **args))

        # Metrics after broken embeddings have been fixed
        model = LinearTrivialDistanceSiamese(feat_dim=embeddings_broken.shape[-1], identity_init=True)
        processor = PairwiseReranker(pairwise_model=model, top_n=100, batch_size=16, num_workers=0)
        prediction_upd = processor.process(pred, dataset=dataset)

        metrics_upd = flatten_dict(calc_retrieval_metrics(retrieved_ids=prediction_upd.retrieved_ids, **args))

        for key in metrics.keys():
            metric = metrics[key]
            metric_upd = metrics_upd[key]
            assert metric_upd >= metric, (key, metric, metric_upd)


class RandomPairwise(IPairwiseModel):
    def __init__(self):  # type: ignore
        super(RandomPairwise, self).__init__()
        self.parameter = torch.nn.Linear(1, 1)

    def forward(self, x1: FloatTensor, x2: FloatTensor) -> FloatTensor:
        return torch.sigmoid(torch.rand(x1.shape[0])).float()

    def predict(self, x1: FloatTensor, x2: FloatTensor) -> FloatTensor:
        return self.forward(x1, x2)


@pytest.mark.long
@pytest.mark.parametrize("top_n", [2, 4, 5])
def test_processing_not_changing_non_sensitive_metrics(top_n: int) -> None:
    # The idea of the test is that postprocessing of first n elements
    # cannot change cmc@n and precision@n

    n_repetitions = 5
    for _ in range(n_repetitions):
        # Let's construct some random input
        dataset = perfect_case()
        embeddings_rand = torch.randn_like(dataset._embeddings).float()

        prediction = RetrievalPrediction.compute_from_embeddings(embeddings=embeddings_rand, dataset=dataset)

        args = {"cmc_top_k": (top_n,), "precision_top_k": (top_n,), "map_top_k": tuple(), "gt_ids": prediction.gt_ids}

        metrics_before = calc_retrieval_metrics(retrieved_ids=prediction.retrieved_ids, **args)

        model = RandomPairwise()
        processor = PairwiseReranker(pairwise_model=model, top_n=top_n, batch_size=4, num_workers=0)
        prediction_upd = processor.process(prediction, dataset=dataset)

        metrics_after = calc_retrieval_metrics(retrieved_ids=prediction_upd.retrieved_ids, **args)

        assert metrics_before == metrics_after
