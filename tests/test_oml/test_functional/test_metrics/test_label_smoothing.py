import pytest
import torch

from oml.functional.label_smoothing import label_smoothing


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
    smoothed = label_smoothing(y, num_classes=num_classes, epsilon=ls)
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
    smoothed = label_smoothing(y, num_classes=num_classes, epsilon=ls, categories=l2c)
    assert torch.allclose(expected_result, smoothed, atol=1e-5)
