import torch

from oml.models.siamese import SiameseL2
from oml.utils.misc_torch import elementwise_dist


def test_initialisation_is_the_same_with_l2_distance() -> None:
    feat_dim = 12
    bs = 8
    x1 = torch.randn(bs, feat_dim)
    x2 = torch.randn(bs, feat_dim)

    dists = elementwise_dist(x1, x2, p=2)

    model = SiameseL2(feat_dim=feat_dim, init_with_identity=True)
    output = model(x1, x2)

    assert torch.isclose(dists, output).all()
