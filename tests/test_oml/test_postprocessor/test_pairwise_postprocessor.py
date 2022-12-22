from typing import Any, Tuple

import pytest
import torch
from torch import Tensor

from oml.functional.metrics import (
    apply_mask_to_ignore,
    calc_distance_matrix,
    calc_mask_to_ignore,
)
from oml.models.siamese import SiameseL2
from oml.postprocessors.pairwise_postprocessor import PairwisePostprocessor
from oml.utils.misc_torch import take_slice_2d


def independent_query_gallery_case() -> Tuple[Tensor, Tensor, Tensor]:
    embeddings = torch.randn((7, 4))
    is_query = torch.tensor([1, 1, 1, 0, 0, 0, 0]).bool()
    is_gallery = torch.tensor([0, 0, 0, 1, 1, 1, 1]).bool()
    return embeddings, is_query, is_gallery


def same_query_gallery_case() -> Tuple[Tensor, Tensor, Tensor]:
    sz = 7
    embeddings = torch.randn((sz, 4))
    is_query = torch.ones(sz).bool()
    is_gallery = torch.ones(sz).bool()
    return embeddings, is_query, is_gallery


@pytest.mark.parametrize("case", [independent_query_gallery_case(), same_query_gallery_case()])
def test_pairwise_postprocessor(case: Any) -> None:
    top_n = 2

    embeddings, is_query, is_gallery = case

    distance_matrix = calc_distance_matrix(embeddings, is_query, is_gallery)
    mask_to_ignore = calc_mask_to_ignore(is_query, is_gallery)
    distance_matrix, _ = apply_mask_to_ignore(distance_matrix, mask_gt=None, mask_to_ignore=mask_to_ignore)

    model = SiameseL2(feat_dim=embeddings.shape[-1], init_with_identity=True)
    processor = PairwisePostprocessor(pairwise_model=model, top_n=top_n)

    output, picked_galleries = processor.process(
        embeddings=embeddings,
        is_query=is_query,
        is_gallery=is_gallery,
        distance_matrix=distance_matrix.clone(),
    )

    assert torch.isclose(output, take_slice_2d(distance_matrix, picked_galleries)).all()
