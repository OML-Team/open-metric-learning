import numpy as np
import torch

from oml.utils.misc_torch import elementwise_dist


def test_elementwise_dist() -> None:
    x1 = torch.randn(size=(3, 4))
    x2 = torch.randn(size=(3, 4))

    val_torch = elementwise_dist(x1=x1, x2=x2, p=2)
    val_custom = np.sqrt(((np.array(x1) - np.array(x2)) ** 2).sum(axis=1))

    assert torch.isclose(val_torch, torch.tensor(val_custom)).all()
