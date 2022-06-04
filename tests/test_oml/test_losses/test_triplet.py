import pytest
import torch
from torch.nn import TripletMarginLoss

from oml.losses.triplet import TripletLoss


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("margin", [0.2, 0.5, 1.5])
def test_triplet_loss_vs_torch_version(reduction: str, margin: float) -> None:
    criterion_torch = TripletMarginLoss(reduction=reduction, margin=margin)
    criterion_our = TripletLoss(reduction=reduction, margin=margin)

    for _ in range(10):
        anchor = torch.randn(32, 1024)
        positive = torch.randn(32, 1024)
        negative = torch.randn(32, 1024)

        x_torch = criterion_torch(anchor, positive, negative)
        x_our = criterion_our(anchor, positive, negative)

        assert x_torch.allclose(x_our, rtol=0.01)


def test_soft_triplet_loss() -> None:
    vec = torch.randn(32, 1024)
    gt = torch.tensor(0.693147180559945)  # this value is log1p(0)

    assert TripletLoss(margin=None, reduction="mean")(vec, vec, vec).isclose(gt)
