import math
from typing import Any, Dict, Tuple, Union

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD

from oml.losses.arcface import ArcFaceLoss
from oml.utils.misc import set_global_seed


@pytest.fixture(scope="session", params=[24, 42])
def dataset1(request: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    set_global_seed(request.param)
    X = torch.rand((10, 2))
    y = (X[:, 0] > X[:, 1]).to(torch.int64)
    return X, y


@pytest.fixture
def dataset2() -> Tuple[torch.Tensor, torch.Tensor]:
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
    return (X, y), {0: 0, 1: 1, 2: 1}


@pytest.fixture
def dataset3() -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [
            [1, 0, 0],
            [0, 0.2, 0.8],
            [0, 0.8, 0.2],
        ]
    )
    y = torch.tensor([0, 1, 2])
    return (X, y), {0: 0, 1: 1, 2: 1}


def arcface_functional_for_comarison(
    x: torch.Tensor,
    label: torch.Tensor,
    weight: Union[torch.Tensor, nn.Parameter],
    criterion: nn.Module,
    m: float = 0.5,
    s: float = 64,
) -> torch.Tensor:
    """
    Implementation from https://github.com/ronghuaiyang/arcface-pytorch

    """

    cos_m = math.cos(m)
    sin_m = math.sin(m)
    th = math.cos(math.pi - m)
    mm = math.sin(math.pi - m) * m

    # --------------------------- cos(theta) & phi(theta) ---------------------------
    cosine = F.linear(F.normalize(x), F.normalize(weight))
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
    phi = cosine * cos_m - sine * sin_m
    phi = torch.where(cosine > th, phi, cosine - mm)
    # --------------------------- convert label to one-hot ---------------------------
    one_hot = torch.zeros(cosine.size())
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    output *= s

    return criterion(output, label)


def check_arcface_trainable(X: Any, y: Any) -> None:
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


def test_arcface_trainable_rand_dataset(dataset1: Tuple[torch.Tensor, torch.Tensor]) -> None:
    X, y = dataset1
    check_arcface_trainable(X, y)


def test_arcface_trainable_ls_dataset(dataset2: Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[int, int]]) -> None:
    (X, y), _ = dataset2
    check_arcface_trainable(X, y)


@pytest.mark.parametrize("label_smoothing", [0.1, 0.2, 0.5])
def test_label_smoothing_working(
    label_smoothing: float, dataset2: Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[int, int]]
) -> None:
    (X, y), label2category = dataset2
    loss = ArcFaceLoss(
        X.shape[-1],
        len(torch.unique(y)),
        label2category=label2category,
        smoothing_epsilon=label_smoothing,
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
def test_label_smoothing_fuzzy(
    label_smoothing: float, dataset3: Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[int, int]]
) -> None:
    (X, y), label2category = dataset3

    loss_ls = ArcFaceLoss(3, 3, label2category=label2category, smoothing_epsilon=label_smoothing)
    loss = ArcFaceLoss(3, 3)

    w = torch.eye(3).float()
    with torch.no_grad():
        loss.weight.copy_(w)
        loss_ls.weight.copy_(w)

    assert loss(X, y).item() > loss_ls(X, y).item()


@pytest.mark.parametrize("label_smoothing", [-1, 1])
def test_label_smoothing_raises_bad_ls(label_smoothing: float) -> None:
    with pytest.raises(AssertionError):
        ArcFaceLoss(1, 1, smoothing_epsilon=label_smoothing, label2category={1: 1})


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("m,s", [(0.4, 64), (0.5, 64), (0.5, 48)])
def test_arface(m: float, s: float, seed: int) -> None:
    set_global_seed(seed)
    loss = ArcFaceLoss(3, 3, m=m, s=s).cpu()

    x = torch.randn((3, 3)).cpu()
    y = torch.tensor([0, 1, 2]).cpu()

    loss_value = loss(x, y)
    loss_value2 = arcface_functional_for_comarison(x, y, weight=loss.weight, criterion=loss.criterion, m=m, s=s)

    assert torch.allclose(loss_value, loss_value2)
