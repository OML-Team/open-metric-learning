import numpy as np
import pytest
import torch

from oml.utils.misc_torch import elementwise_dist, label_smoothing


def test_elementwise_dist() -> None:
    x1 = torch.randn(size=(3, 4))
    x2 = torch.randn(size=(3, 4))

    val_torch = elementwise_dist(x1=x1, x2=x2, p=2)
    val_custom = np.sqrt(((np.array(x1) - np.array(x2)) ** 2).sum(axis=1))

    assert torch.isclose(val_torch, torch.tensor(val_custom)).all()


@pytest.mark.parametrize(
    ["num_classes", "ls", "expected_result"],
    [
        (2, 0, torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
        (2, 0.1, torch.tensor([[0.05, 0.95], [0.95, 0.05]])),
        (3, 0.3, torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]])),
    ],
)
def test_label_smoothing(num_classes: int, ls: float, expected_result: torch.Tensor) -> None:
    y = torch.tensor([1, 0])
    smoothed = label_smoothing(y, num_classes=num_classes, label_smoothing=ls)
    assert torch.allclose(expected_result, smoothed, atol=1e-5)


@pytest.mark.parametrize(
    ["num_classes", "ls", "expected_result", "l2c"],
    [
        (3, 0.2, torch.tensor([[0.0, 0.9, 0.1], [1.0, 0.0, 0.0]]), torch.tensor([0, 1, 1])),
        (3, 0.2, torch.tensor([[0.1, 0.9, 0.0], [0.9, 0.1, 0.0]]), torch.tensor([0, 0, 1])),
    ],
)
def test_label_smoothing_with_l2c(
    num_classes: int, ls: float, expected_result: torch.Tensor, l2c: torch.Tensor
) -> None:
    y = torch.tensor([1, 0])
    smoothed = label_smoothing(y, num_classes=num_classes, label_smoothing=ls, label2category=l2c)
    assert torch.allclose(expected_result, smoothed, atol=1e-5)
