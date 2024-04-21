from functools import partial
from random import randint, random
from typing import Tuple

import pytest
import torch
from torch import Tensor

from oml.functional.metrics import calc_retrieval_metrics
from oml.interfaces.datasets import IQueryGalleryDataset, IQueryGalleryLabeledDataset
from oml.interfaces.models import IPairwiseModel
from oml.models.meta.siamese import LinearTrivialDistanceSiamese
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.misc import flatten_dict, one_hot, set_global_seed
from oml.utils.misc_torch import normalise, pairwise_dist
from tests.test_integrations.utils import (
    EmbeddingsQueryGalleryDataset,
    EmbeddingsQueryGalleryLabeledDataset,
)

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
@pytest.mark.parametrize("pairwise_distances_bias", [0, -100, +100])
@pytest.mark.parametrize("fixture_name", ["independent_query_gallery_case", "shared_query_gallery_case"])
def test_trivial_processing_does_not_change_distances_order(
    request: pytest.FixtureRequest, fixture_name: str, top_n: int, pairwise_distances_bias: float
) -> None:
    for _ in range(10):
        dataset, embeddings = request.getfixturevalue(fixture_name)

        distances = pairwise_dist(x1=embeddings[dataset.get_query_ids()], x2=embeddings[dataset.get_gallery_ids()], p=2)

        print(distances, "zzzz")

        model = LinearTrivialDistanceSiamese(
            embeddings.shape[-1], output_bias=pairwise_distances_bias, identity_init=True
        )
        processor = PairwiseReranker(pairwise_model=model, top_n=top_n, num_workers=0, batch_size=64)

        distances_processed = processor.process(distances=distances.clone(), dataset=dataset)

        assert (distances_processed.argsort() == distances.argsort()).all()


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
        distances = pairwise_dist(embeddings[dataset.get_query_ids()], embeddings[dataset.get_gallery_ids()], p=2)

        labels_q = torch.tensor(dataset.get_labels()[dataset.get_query_ids()])
        labels_g = torch.tensor(dataset.get_labels()[dataset.get_gallery_ids()])
        mask_gt = labels_q.unsqueeze(-1) == labels_g

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
        model = LinearTrivialDistanceSiamese(
            feat_dim=embeddings.shape[-1], identity_init=True, output_bias=pairwise_distances_bias
        )
        processor = PairwiseReranker(pairwise_model=model, top_n=top_n, batch_size=16, num_workers=0)
        distances_upd = processor.process(distances, dataset)
        metrics_upd = flatten_dict(calc_retrieval_metrics(distances=distances_upd, **args))

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

    # let's get some random inputs
    dataset, embeddings = perfect_case()
    embeddings = torch.randn_like(embeddings).float()

    top_n = min(top_n, embeddings.shape[1])

    distances = pairwise_dist(embeddings[dataset.get_query_ids()], embeddings[dataset.get_gallery_ids()], p=2)

    labels_q = torch.tensor(dataset.get_labels()[dataset.get_query_ids()])
    labels_g = torch.tensor(dataset.get_labels()[dataset.get_gallery_ids()])
    mask_gt = labels_q.unsqueeze(-1) == labels_g

    args = {
        "cmc_top_k": (top_n,),
        "precision_top_k": (top_n,),
        "fmr_vals": tuple(),
        "map_top_k": tuple(),
        "mask_gt": mask_gt,
    }

    metrics_before = calc_retrieval_metrics(distances=distances, **args)
    ii_closest_before = torch.argsort(distances)

    model = RandomPairwise()
    processor = PairwiseReranker(pairwise_model=model, top_n=top_n, batch_size=4, num_workers=0)
    distances_upd = processor.process(distances=distances, dataset=dataset)

    metrics_after = calc_retrieval_metrics(distances=distances_upd, **args)
    ii_closest_after = torch.argsort(distances_upd)

    assert metrics_before == metrics_after

    # also check that we only re-ranked the first top_n items
    assert (ii_closest_before[:, :top_n] != ii_closest_after[:, :top_n]).any()
    assert (ii_closest_before[:, top_n:] == ii_closest_after[:, top_n:]).all()
