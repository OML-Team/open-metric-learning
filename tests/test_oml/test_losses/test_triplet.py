import pytest
import torch
from torch.nn import TripletMarginLoss

from oml.losses.triplet import TripletLoss


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("margin", [0.2, 0.5, 1.5])
def test_triplet_loss_vs_torch_version(reduction: str, margin: float) -> None:
    criterion_torch = TripletMarginLoss(reduction=reduction, margin=margin)
    criterion_our = TripletLoss(reduction=reduction, margin=margin, need_logs=True)

    anchor = 1 * torch.ones(32, 1024)
    positive = 2 * torch.ones(32, 1024)
    negative = 3 * torch.ones(32, 1024)

    x_torch = criterion_torch(anchor, positive, negative)
    x_our = criterion_our(anchor, positive, negative)

    assert x_torch.allclose(x_our, rtol=0.01)


def test_soft_triplet_loss() -> None:
    vec = torch.randn(32, 1024)
    gt = torch.tensor(0.693147180559945)  # this value is log1p(0)

    assert TripletLoss(margin=None, reduction="mean", need_logs=True)(vec, vec, vec).isclose(gt)
