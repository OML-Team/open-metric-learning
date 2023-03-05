from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from oml.interfaces.models import IExtractor
from oml.models.projection import ExtractorWithMLP
from oml.models.resnet import ResnetExtractor
from oml.models.siamese import ConcatSiamese
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

    # 3. Test pretrained
    for weights in constructor.pretrained_models.keys():
        if ("vitl" in weights) and not download_large_checkpoints:
            continue
        net = constructor.from_pretrained(weights=weights)
        net(torch.randn(1, 3, 224, 224))

    if constructor == ResnetExtractor:
        net = ResnetExtractor.from_pretrained("resnet50_default")
        net(torch.randn(1, 3, 224, 224))

        net = ResnetExtractor.from_pretrained("resnet18_imagenet_v1")
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
    extractor = ExtractorWithMLP(extractor=net, mlp_features=[128]).eval()
    features1 = extractor(im)

    fname = f"{default_arch}_with_mlp_random.pth"
    torch.save({"state_dict": extractor.state_dict()}, fname)
    net = constructor(weights=None, arch=default_arch, **args)
    extractor = ExtractorWithMLP(extractor=net, mlp_features=[128], weights=fname).eval()
    Path(fname).unlink()
    features2 = extractor(im)

    assert torch.allclose(features1, features2)


@pytest.mark.parametrize(
    "constructor,args,default_arch",
    [
        (ViTExtractor, vit_args, "vits8"),
        (ResnetExtractor, resnet_args, "resnet50"),
    ],
)
def test_concat_siamese(constructor: IExtractor, args: Dict[str, Any], default_arch: str) -> None:
    im1 = torch.randn(2, 3, 32, 32)
    im2 = torch.randn(2, 3, 32, 32)

    net = constructor(weights=None, arch=default_arch, **args)
    extractor = ConcatSiamese(extractor=net, mlp_hidden_dims=[128, 10]).eval()
    output1 = extractor(im1, im2)
    assert output1.ndim == 1

    fname = f"{default_arch}_siamese_random.pth"
    torch.save({"state_dict": extractor.state_dict()}, fname)
    net = constructor(weights=None, arch=default_arch, **args)
    extractor = ConcatSiamese(extractor=net, mlp_hidden_dims=[128, 10], weights=fname).eval()
    Path(fname).unlink()
    output2 = extractor(im1, im2)

    assert output2.ndim == 1

    assert torch.allclose(output1, output2)
