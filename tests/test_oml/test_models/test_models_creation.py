import os
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from oml.interfaces.models import IExtractor, IPairwiseModel
from oml.models.meta.projection import ExtractorWithMLP
from oml.models.meta.siamese import ConcatSiamese
from oml.models.resnet.extractor import ResnetExtractor
from oml.models.vit_clip.extractor import ViTCLIPExtractor
from oml.models.vit_dino.extractor import ViTExtractor
from oml.models.vit_dinov2.extractor import ViTExtractor_v2
from oml.models.vit_unicom.extractor import ViTUnicomExtractor
from oml.registry import EXTRACTORS_REGISTRY

SKIP_LARGE_CKPT = True
LARGE_CKPT_NAMES = ["vitl", "resnet101", "resnet152"]

vit_args = {"normalise_features": False, "use_multi_scale": False, "arch": "vits16"}


# todo: add another test where Lightning saves the model
@pytest.mark.parametrize(
    "constructor,args",
    [
        (ViTExtractor, vit_args),
        (ViTExtractor_v2, {"normalise_features": False, "arch": "vits14"}),
        (ViTCLIPExtractor, {"normalise_features": False, "arch": "vitb32_224"}),
        (ViTUnicomExtractor, {"normalise_features": False, "arch": "vitb32_unicom"}),
        (ResnetExtractor, {"normalise_features": True, "gem_p": 7.0, "remove_fc": True, "arch": "resnet50"}),
        (ResnetExtractor, {"normalise_features": True, "gem_p": 7.0, "remove_fc": False, "arch": "resnet50"}),
        (ExtractorWithMLP, {"extractor": ViTExtractor(None, **vit_args), "mlp_features": [128]}),  # type: ignore
    ],
)
def test_extractor(constructor: IExtractor, args: Dict[str, Any]) -> None:
    im = torch.randn(1, 3, 224, 224)

    # 1. Random weights
    extractor = constructor(weights=None, **args).eval()
    features1 = extractor.extract(im)

    # 2. Save random weights to the file, load & compare inference
    fname = "weights_tmp.pth"
    torch.save({"state_dict": extractor.state_dict()}, fname)
    extractor = constructor(weights=fname, **args).eval()
    features2 = extractor.extract(im)
    Path(fname).unlink()

    assert features1.ndim == 2
    assert features1.shape[-1] == extractor.feat_dim
    assert torch.allclose(features1, features2)


@pytest.mark.long
@pytest.mark.skipif(os.getenv("DOWNLOAD_ZOO_IN_TESTS") != "yes", reason="It's a traffic consuming test.")
@pytest.mark.parametrize("constructor", list(EXTRACTORS_REGISTRY.values()))
def test_checkpoints_from_zoo(constructor: IExtractor) -> None:
    im = torch.randn(1, 3, 224, 224)

    for weights in constructor.pretrained_models.keys():
        if any([nm in weights for nm in LARGE_CKPT_NAMES]) and SKIP_LARGE_CKPT:
            continue

        extractor = constructor.from_pretrained(weights=weights).eval()
        extractor.extract(im)

    assert True


@pytest.mark.parametrize(
    "constructor,args",
    [
        (ConcatSiamese, {"extractor": ViTExtractor(None, **vit_args), "mlp_hidden_dims": [128, 10]}),  # type: ignore
    ],
)
def test_pairwise(constructor: IPairwiseModel, args: Dict[str, Any]) -> None:
    im1 = torch.randn(2, 3, 32, 32)
    im2 = torch.randn(2, 3, 32, 32)

    model_pairwise = constructor(weights=None, **args).eval()
    output1 = model_pairwise.predict(im1, im2)
    assert output1.ndim == 1

    fname = "weights_tmp.ckpt"
    torch.save({"state_dict": model_pairwise.state_dict()}, fname)
    model_pairwise = constructor(weights=fname, **args).eval()
    Path(fname).unlink()
    output2 = model_pairwise.predict(im1, im2)

    assert output2.ndim == 1
    assert torch.allclose(output1, output2)
