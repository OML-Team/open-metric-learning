from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from oml.interfaces.models import IExtractor
from oml.models.resnet import ResnetExtractor
from oml.models.vit.clip import ViTCLIPExtractor
from oml.models.vit.vit import ViTExtractor

vit_args = {"normalise_features": False, "use_multi_scale": False, "strict_load": True}
vit_clip_args = {"normalise_features": False, "strict_load": True}
resnet_args = {"normalise_features": True, "gem_p": 7.0, "remove_fc": False, "strict_load": True}


@pytest.mark.parametrize(
    "constructor,args,default_arch",
    [
        (ViTExtractor, vit_args, "vits16"),
        (ViTCLIPExtractor, vit_clip_args, "vitb32_224"),
        (ResnetExtractor, resnet_args, "resnet50"),
    ],
)
def test_creation(constructor: IExtractor, args: Dict[str, Any], default_arch: str) -> None:
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
            if not key.endswith("224"):
                continue
            arch = key.lstrip("sber_").lstrip("openai_")
        else:
            arch = key.split("_")[0]

        print(constructor, key, arch, args)
        net = constructor(weights=key, arch=arch, **args)
        net(torch.randn(1, 3, 224, 224))

    assert True
