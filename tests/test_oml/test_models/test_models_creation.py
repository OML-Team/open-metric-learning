from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from oml.interfaces.models import IExtractor
from oml.models.projection import ExtractorWithMLP
from oml.models.resnet import ResnetExtractor
from oml.models.vit.clip import ViTCLIPExtractor
from oml.models.vit.vit import ViTExtractor

vit_args = {"normalise_features": False, "use_multi_scale": False, "strict_load": True}
vit_clip_args = {"normalise_features": False, "strict_load": True}
resnet_args = {"normalise_features": True, "gem_p": 7.0, "remove_fc": False, "strict_load": True}


@pytest.mark.parametrize(
    "constructor,args,default_arch,download_large_checkpoints",
    [
        (ViTExtractor, vit_args, "vits16", False),
        (ViTCLIPExtractor, vit_clip_args, "vitb32_224", False),
        (ResnetExtractor, resnet_args, "resnet50", False),
    ],
)
def test_creation(
    constructor: IExtractor, args: Dict[str, Any], default_arch: str, download_large_checkpoints: bool
) -> None:
    # 1. Random weights
    net = constructor(weights=None, arch=default_arch, **args)
    net.forward(torch.randn(1, 3, 224, 224))

    # 2. Load from file
    fname = f"{default_arch}_random.pth"
    torch.save({"state_dict": net.state_dict()}, fname)
    net = constructor(weights=fname, arch=default_arch, **args)
    net(torch.randn(1, 3, 224, 224))
    Path(fname).unlink()

    # 3. Pretrained checkpoints
    for key in constructor.pretrained_models.keys():
        if constructor == ViTCLIPExtractor:
            is_large_model = "vitl" in key
            if is_large_model and not download_large_checkpoints:
                continue
            arch = key.split("_", maxsplit=1)[1]  # sber_vitb16_224 -> vitb16_224, openai_vitb16_224 -> vitb16_224
        else:
            arch = key.split("_")[0]

        net = constructor(weights=key, arch=arch, **args)
        net(torch.randn(1, 3, 224, 224))

    assert True


@pytest.mark.parametrize(
    "constructor,args,default_arch",
    [
        (ViTExtractor, vit_args, "vits16"),
        (ViTCLIPExtractor, vit_clip_args, "vitb32_224"),
        (ResnetExtractor, resnet_args, "resnet50"),
    ],
)
def test_extractor_with_mlp(constructor: IExtractor, args: Dict[str, Any], default_arch: str) -> None:
    im = torch.randn(1, 3, 224, 224)

    net = constructor(weights=None, arch=default_arch, **args)
    extractor = ExtractorWithMLP(extractor=net, mlp_features=[128])
    features1 = extractor(im)

    fname = f"{default_arch}_with_mlp_random.pth"
    torch.save({"state_dict": extractor.state_dict()}, fname)
    net = constructor(weights=None, arch=default_arch, **args)
    extractor = ExtractorWithMLP(extractor=net, mlp_features=[128], weights=fname)
    Path(fname).unlink()
    features2 = extractor(im)

    assert torch.allclose(features1, features2)
