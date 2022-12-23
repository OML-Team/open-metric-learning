from typing import Tuple

import pytest
import torch
from torch import Tensor

from oml.functional.metrics import (
    apply_mask_to_ignore,
    calc_distance_matrix,
    calc_mask_to_ignore,
)
from oml.models.siamese import SimpleSiamese
from oml.postprocessors.pairwise_embeddings import PairwiseEmbeddingsPostprocessor


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


@pytest.mark.parametrize("top_n", [1, 2, 5, 1000])
@pytest.mark.parametrize("fixture_name", ["independent_query_gallery_case", "same_query_gallery_case"])
def test_identity_processing(request: pytest.FixtureRequest, fixture_name: str, top_n: int) -> None:
    embeddings, is_query, is_gallery = request.getfixturevalue(fixture_name)
    embeddings_query = embeddings[is_query]
    embeddings_gallery = embeddings[is_gallery]

    distances = calc_distance_matrix(embeddings, is_query, is_gallery)
    mask_to_ignore = calc_mask_to_ignore(is_query, is_gallery)
    distances, _ = apply_mask_to_ignore(distances, mask_gt=None, mask_to_ignore=mask_to_ignore)

    id_model = SimpleSiamese(feat_dim=embeddings.shape[-1], identity_init=True)
    id_processor = PairwiseEmbeddingsPostprocessor(pairwise_model=id_model, top_n=top_n)

    distances_processed = id_processor.process(
        emb_query=embeddings_query,
        emb_gallery=embeddings_gallery,
        distances=distances.clone(),
    )

    assert torch.isclose(distances, distances_processed).all()
