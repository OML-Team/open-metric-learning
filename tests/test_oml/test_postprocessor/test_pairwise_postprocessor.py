import pytest
import torch

from oml.functional.metrics import calc_distance_matrix, calc_mask_to_ignore
from oml.models.siamese import SiameseL2
from oml.postprocessors.pairwise_postprocessor import PairwisePostprocessor
from oml.utils.misc_torch import elementwise_dist


def independent_query_gallery_case():
    embeddings = torch.randn((7, 4))
    is_query = torch.tensor([1, 1, 1, 0, 0, 0, 0]).bool()
    is_gallery = torch.tensor([0, 0, 0, 1, 1, 1, 1]).bool()
    return embeddings, is_query, is_gallery


def same_query_gallery_case():
    sz = 7
    embeddings = torch.randn((sz, 4))
    is_query = torch.ones(sz).bool()
    is_gallery = torch.ones(sz).bool()
    return embeddings, is_query, is_gallery


@pytest.mark.parametrize("case", [independent_query_gallery_case(), same_query_gallery_case()])
def test_pairwise_postprocessor(case):
    top_n = 2

    embeddings, is_query, is_gallery = case

    distance_matrix = calc_distance_matrix(embeddings, is_query, is_gallery)
    mask_to_ignore = calc_mask_to_ignore(is_query, is_gallery)
    distance_matrix[mask_to_ignore] = float("inf")  # todo

    model = SiameseL2(feat_dim=embeddings.shape[-1], init_with_identity=True)
    processor = PairwisePostprocessor(pairwise_model=model, top_n=top_n)
    output, picked_galleries = processor.process(
        embeddings=embeddings,
        is_query=is_query,
        is_gallery=is_gallery,
        mask_to_ignore=mask_to_ignore,
        distance_matrix=distance_matrix.clone()
    )

    distance_matrix = distance_matrix.sort(dim=-1)[0][:, :top_n]

    assert torch.isclose(output, distance_matrix).all()

    for i_q, ii_gallery in enumerate(picked_galleries):
        for j_g, i_gallery in enumerate(ii_gallery):
            x1 = embeddings[i_q].unsqueeze(0)
            x2 = embeddings[i_gallery].unsqueeze(0)
            assert torch.isclose(elementwise_dist(x1, x2), output[i_q, j_g])


labels = torch.tensor([9, 1, 2, 3, 4])

picked = torch.tensor([[0, 1], [1, 2]]).long()

print(labels[picked] == torch.tensor([9, 1]))