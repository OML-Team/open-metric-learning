from functools import partial
from math import sqrt

import torch

from oml.datasets.embeddings import EmbeddingsQueryGalleryDataset
from oml.retrieval.prediction import RetrievalPrediction
from oml.utils.misc import one_hot

FEAT_DIM = 8
oh = partial(one_hot, dim=FEAT_DIM)


def test_prediction() -> None:
    embeddings = torch.stack([oh(0), oh(1), oh(1), oh(0), oh(1), oh(2)]).squeeze().float()

    dataset = EmbeddingsQueryGalleryDataset(
        embeddings=embeddings,
        labels=torch.tensor([0, 1, 1, 0, 1, 2]).long(),
        is_query=torch.tensor([True, True, True, False, False, False]).bool(),
        is_gallery=torch.tensor([False, False, False, True, True, True]).bool(),
    )

    pred = RetrievalPrediction.compute_from_embeddings(embeddings=embeddings, dataset=dataset, n_items_to_retrieve=5)

    retrieved_ids_expected = torch.tensor([[0, 1, 2], [1, 0, 2], [1, 0, 2]]).long()

    distances_expected = torch.tensor([[0, sqrt(2), sqrt(2)], [0, sqrt(2), sqrt(2)], [0, sqrt(2), sqrt(2)]])

    gt_ids_expected = [torch.tensor([0]), torch.tensor([1]), torch.tensor([1])]

    assert (retrieved_ids_expected == pred.retrieved_ids).all()
    assert all([(x1 == x2).all() for (x1, x2) in zip(gt_ids_expected, pred.gt_ids)])
    assert torch.isclose(distances_expected, pred.distances).all()


# todo 522: add test that checks if there are no gt_ids we fail (because i rm the old test from retrieval metrics)
