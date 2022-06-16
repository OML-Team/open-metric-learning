import pytest
import torch
from torch.nn import CrossEntropyLoss

from oml.losses.arcface import ArcFace
from oml.models.vit.vit import ViTExtractor


@pytest.mark.parametrize("criterion_fun", [ArcFace, CrossEntropyLoss])
def test_train_with_cls_losses(criterion_fun) -> None:  # type: ignore
    n_cls = 2
    bs = 6

    criterion = criterion_fun()

    model = ViTExtractor(
        weights="random",
        arch="vits8",
        normalise_features=False,
        use_multi_scale=False,
        strict_load=False,
        out_dim=n_cls,
    )

    inp = torch.randn(size=(bs, 3, 8, 8))
    label = torch.randint(low=0, high=n_cls - 1, size=(bs,))

    out = model(inp)
    criterion(out, label)

    assert True
