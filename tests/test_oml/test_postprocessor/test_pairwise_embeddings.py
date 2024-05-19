import random
from functools import partial
from random import randint
from typing import Tuple

import pytest
import torch
from torch import Tensor

from oml.interfaces.datasets import IQueryGalleryDataset, IQueryGalleryLabeledDataset
from oml.interfaces.models import IPairwiseModel
from oml.metrics.embeddings import calc_retrieval_metrics_rr
from oml.models.meta.siamese import LinearTrivialDistanceSiamese
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.retrieval.retrieval_results import RetrievalResults
from oml.utils.misc import flatten_dict, one_hot, set_global_seed
from oml.utils.misc_torch import normalise
from tests.test_integrations.utils import (
    EmbeddingsQueryGalleryDataset,
    EmbeddingsQueryGalleryLabeledDataset,
)
from tests.utils import check_if_sequence_of_tensors_are_equal

FEAT_SIZE = 8
oh = partial(one_hot, dim=FEAT_SIZE)


@pytest.fixture
def independent_query_gallery_case() -> Tuple[IQueryGalleryDataset, Tensor]:
    sz = 7
    feat_dim = 12

    is_query = torch.ones(sz).bool()
    is_query[: sz // 2] = False

    is_gallery = torch.ones(sz).bool()
    is_gallery[sz // 2 :] = False

    embeddings = normalise(torch.randn((sz, feat_dim))).float()

    dataset = EmbeddingsQueryGalleryDataset(embeddings=embeddings, is_query=is_query, is_gallery=is_gallery)

    embeddings_inference = embeddings.clone()  # pretend it's our inference results

    return dataset, embeddings_inference


@pytest.fixture
def shared_query_gallery_case() -> Tuple[IQueryGalleryDataset, Tensor]:
    sz = 7
    feat_dim = 4

    embeddings = normalise(torch.randn((sz, feat_dim))).float()

    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=embeddings, is_query=torch.ones(sz).bool(), is_gallery=torch.ones(sz).bool()
    )

    embeddings_inference = embeddings.clone()  # pretend it's our inference results

    return dataset, embeddings_inference


@pytest.mark.long
@pytest.mark.parametrize("top_n", [2, 5, 100])
@pytest.mark.parametrize("pairwise_distances_bias", [0, -5, +5])
@pytest.mark.parametrize("fixture_name", ["independent_query_gallery_case", "shared_query_gallery_case"])
def test_trivial_processing_does_not_change_distances_order(
    request: pytest.FixtureRequest, fixture_name: str, top_n: int, pairwise_distances_bias: float
) -> None:
    dataset, embeddings = request.getfixturevalue(fixture_name)

    rr = RetrievalResults.compute_from_embeddings(embeddings, dataset, n_items_to_retrieve=150)

    model = LinearTrivialDistanceSiamese(embeddings.shape[-1], output_bias=pairwise_distances_bias, identity_init=True)
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, num_workers=0, batch_size=64)

    rr_upd = processor.process(rr, dataset=dataset)

    assert check_if_sequence_of_tensors_are_equal(rr.retrieved_ids, rr_upd.retrieved_ids)

    if pairwise_distances_bias == 0:
        assert check_if_sequence_of_tensors_are_equal(rr.distances, rr_upd.distances)
    else:
        assert not check_if_sequence_of_tensors_are_equal(rr.distances, rr_upd.distances)


def perfect_case() -> Tuple[IQueryGalleryLabeledDataset, Tensor]:
    embeddings = torch.stack([oh(1), oh(2), oh(3), oh(1), oh(2), oh(1), oh(2), oh(3)]).float()

    dataset = EmbeddingsQueryGalleryLabeledDataset(
        embeddings=embeddings,
        labels=torch.tensor([1, 2, 3, 1, 2, 1, 2, 3]).long(),
        is_query=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).bool(),
        is_gallery=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).bool(),
    )

    embeddings_inference = embeddings.clone()

    return dataset, embeddings_inference


@pytest.mark.long
@pytest.mark.parametrize("pairwise_distances_bias", [0, -100, +100])
def test_trivial_processing_fixes_broken_perfect_case(pairwise_distances_bias: float) -> None:
    """
    The idea of the test is the following:

    1. We generate perfect set of labels and one-hot embeddings representing them
    2. We make distances matrix broken by randomly replacing some distance with a random value
    3. We apply pairwise postprocessor which simply restores l2 distances in the distances matrix
    4. No matter what other parameters were, metrics must be the same as before or become better

    """
    n_repetitions = 20
    for _ in range(n_repetitions):

        dataset, embeddings = perfect_case()
        rr = RetrievalResults.compute_from_embeddings(embeddings.float(), dataset, n_items_to_retrieve=100)

        nq = len(rr.distances)
        ng = len(rr.distances[0])

        # Let's randomly break the case
        for _ in range(5):
            iq = random.randint(0, nq - 1)
            perm = torch.randperm(len(rr.retrieved_ids))
            rr.retrieved_ids[iq][:] = rr.retrieved_ids[iq][perm]

        # As mentioned before, for this test the exact values of parameters don't matter
        top_k = (randint(1, ng - 1),)
        top_n = randint(2, 10)

        args = {"precision_top_k": top_k, "map_top_k": top_k, "cmc_top_k": top_k}

        # Metrics before
        metrics = flatten_dict(calc_retrieval_metrics_rr(rr, **args))  # type: ignore

        # Metrics after broken distances have been fixed
        model = LinearTrivialDistanceSiamese(
            feat_dim=embeddings.shape[-1], identity_init=True, output_bias=pairwise_distances_bias
        )
        processor = PairwiseReranker(pairwise_model=model, top_n=top_n, batch_size=16, num_workers=0)
        rr_upd = processor.process(rr, dataset)
        metrics_upd = flatten_dict(calc_retrieval_metrics_rr(rr_upd, **args))  # type: ignore

        for key in metrics.keys():
            metric = metrics[key]
            metric_upd = metrics_upd[key]
            assert metric_upd >= metric, (key, metric, metric_upd)


class RandomPairwise(IPairwiseModel):
    def __init__(self):  # type: ignore
        super(RandomPairwise, self).__init__()
        self.parameter = torch.nn.Linear(1, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return torch.sigmoid(torch.rand(x1.shape[0]))

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1, x2)


@pytest.mark.long
@pytest.mark.parametrize("top_n", [2, 4, 5])
def test_processing_not_changing_non_sensitive_metrics(top_n: int) -> None:
    # The idea of the test is that postprocessing of first n elements
    # cannot change cmc@n and precision@n

    set_global_seed(42)

    dataset, embeddings = perfect_case()

    top_n = min(top_n, embeddings.shape[1])

    rr = RetrievalResults.compute_from_embeddings(embeddings, dataset, n_items_to_retrieve=100)

    args = {
        "cmc_top_k": (top_n,),
        "precision_top_k": (top_n,),
        "map_top_k": tuple(),
    }

    metrics_before = calc_retrieval_metrics_rr(rr, **args)  # type: ignore

    model = RandomPairwise()
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, batch_size=4, num_workers=0)
    rr_upd = processor.process(rr, dataset=dataset)

    metrics_after = calc_retrieval_metrics_rr(rr_upd, **args)  # type: ignore

    assert metrics_before == metrics_after

    top_ids = [r[:top_n] for r in rr.retrieved_ids]
    top_ids_upd = [r[:top_n] for r in rr_upd.retrieved_ids]
    assert not check_if_sequence_of_tensors_are_equal(top_ids, top_ids_upd)

    last_ids = [r[top_n:] for r in rr.retrieved_ids]
    last_ids_upd = [r[top_n:] for r in rr_upd.retrieved_ids]
    assert check_if_sequence_of_tensors_are_equal(last_ids, last_ids_upd)
