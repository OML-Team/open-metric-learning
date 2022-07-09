from pathlib import Path
from typing import Optional, Union

import pytest
import torch

from oml.registry.models import get_extractor


@pytest.mark.parametrize("weights,arch", [("pretrained_dino", "vits8"), ("random", "vitb8")])
@pytest.mark.parametrize("normalise_features", [False, True])
@pytest.mark.parametrize("use_multi_scale", [False, True])
def test_vit_creation(weights: Union[str, Path], arch: str, normalise_features: bool, use_multi_scale: bool) -> None:
    kwargs = {
        "weights": weights,
        "arch": arch,
        "normalise_features": normalise_features,
        "use_multi_scale": use_multi_scale,
        "strict_load": True,
    }

    extractor = get_extractor("vit", kwargs)
    extractor.eval()

    extractor.extract(torch.randn(1, 3, 360, 360))

    assert True


@pytest.mark.parametrize("hid_dim,out_dim,remove_fc,strict_load", [(None, None, True, False), (512, 128, False, False)])
def test_resnet_creation(hid_dim: Optional[int], out_dim: Optional[int], remove_fc: bool, strict_load: bool) -> None:
    kwargs = {
        "weights": "pretrained",
        "arch": "resnet50",
        "hid_dim": hid_dim,
        "out_dim": out_dim,
        "normalise_features": False,
        "gem_p": 7.0,
        "remove_fc": remove_fc,
        "strict_load": strict_load,
    }

    extractor = get_extractor("resnet", kwargs)
    extractor.eval()

    extractor.extract(torch.randn(1, 3, 360, 360))

    assert True
