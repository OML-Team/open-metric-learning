from typing import Callable, Dict, Tuple

import pytest
import torch
from torch.optim import SGD

from oml.losses.arcface import ArcFaceLoss


def dataset1(seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X = torch.rand((10, 2))
    y = (X[:, 0] > X[:, 1]).to(torch.int64)
    return X, y


def dataset2(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    seed
    X = torch.tensor(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0.5, 0.5],
            [0, 0.5, 0.5],
        ]
    )
    y = torch.tensor([2, 1, 0, 1, 2])
    return X, y


@pytest.fixture
def dataset2_label2category() -> Dict[int, int]:
    return {0: 0, 1: 1, 2: 1}


@pytest.mark.parametrize("seed", [2, 4, 8, 16])
@pytest.mark.parametrize("dataset,dataset_seed", [(dataset1, 24), (dataset1, 42), (dataset2, 0)])
def test_arcface_trainable(
    seed: int, dataset: Callable[[int], Tuple[torch.Tensor, torch.Tensor]], dataset_seed: int
) -> None:
    torch.manual_seed(seed)

    X, y = dataset(dataset_seed)
    loss = ArcFaceLoss(X.shape[-1], len(torch.unique(y)))

    l0 = loss(X, y).item()
    sgd = SGD(params=loss.parameters(), lr=0.05, momentum=0.5)

    for _ in range(10):
        sgd.zero_grad()
        lc = loss(X, y)
        lc.backward()
        sgd.step()

    l1 = loss(X, y).item()
    assert l0 > l1


@pytest.mark.parametrize("label_smoothing", [0.1, 0.2, 0.5])
def test_label_smoothing_working(label_smoothing: float, dataset2_label2category: Dict[int, int]) -> None:
    X, y = dataset2()
    loss = ArcFaceLoss(
        X.shape[-1],
        len(torch.unique(y)),
        label2category=dataset2_label2category,
        label_smoothing=label_smoothing,
    )

    l0 = loss(X, y).item()
    sgd = SGD(params=loss.parameters(), lr=0.05, momentum=0.5)

    for _ in range(10):
        sgd.zero_grad()
        lc = loss(X, y)
        lc.backward()
        sgd.step()

    l1 = loss(X, y).item()
    assert l0 > l1


@pytest.mark.parametrize("label_smoothing", [0.1, 0.5])
def test_label_smoothing_fuzzy(label_smoothing: float) -> None:
    X = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    y = torch.tensor([0, 1, 2])

    loss_ls = ArcFaceLoss(3, 3, label2category={0: 0, 1: 1, 2: 1}, label_smoothing=label_smoothing)
    loss = ArcFaceLoss(3, 3)

    w = torch.eye(3).float()
    with torch.no_grad():
        loss.weight.copy_(w)
        loss_ls.weight.copy_(w)

    assert loss(X, y).item() > loss_ls(X, y).item()


def test_label_smoothing_raises_no_l2c() -> None:
    with pytest.raises(AssertionError):
        ArcFaceLoss(1, 1, label_smoothing=0.3)


@pytest.mark.parametrize("label_smoothing", [0, -1, 1])
def test_label_smoothing_raises_bad_ls(label_smoothing: float) -> None:
    with pytest.raises(AssertionError):
        ArcFaceLoss(1, 1, label_smoothing=label_smoothing, label2category={1: 1})
