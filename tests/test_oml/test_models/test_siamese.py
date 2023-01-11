import pytest
import torch

from oml.models.siamese import VectorsSiamese
from oml.utils.misc_torch import elementwise_dist


@pytest.mark.parametrize("bs", [32, 4, 1])
@pytest.mark.parametrize("feat_dim", [8, 1024])
def test_simple_siamese_identity_initialisation(feat_dim: int, bs: int) -> None:
    x1 = torch.randn(bs, feat_dim)
    x2 = torch.randn(bs, feat_dim)

    distances = elementwise_dist(x1=x1, x2=x2, p=2)

    model = VectorsSiamese(feat_dim=feat_dim, identity_init=True)
    distances_estimated = model(x1=x1, x2=x2)

    assert torch.isclose(distances, distances_estimated).all()
