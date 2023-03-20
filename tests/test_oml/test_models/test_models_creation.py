from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from oml.interfaces.models import IExtractor, IPairwiseModel
from oml.models.meta.projection import ExtractorWithMLP
from oml.models.meta.siamese import ConcatSiamese
from oml.models.resnet import ResnetExtractor
from oml.models.vit.clip import ViTCLIPExtractor
from oml.models.vit.vit import ViTExtractor

SKIP_LARGE_CKPT = True

vit_args = {"normalise_features": False, "use_multi_scale": False, "arch": "vits16"}


@pytest.mark.parametrize(
    "constructor,args",
    [
        (ViTExtractor, vit_args),
        (ViTCLIPExtractor, {"normalise_features": False, "arch": "vitb32_224"}),
        (ResnetExtractor, {"normalise_features": True, "gem_p": 7.0, "remove_fc": False, "arch": "resnet50"}),
        (ExtractorWithMLP, {"extractor": ViTExtractor(None, **vit_args), "mlp_features": [128]}),  # type: ignore
    ],
)
def test_extractor(constructor: IExtractor, args: Dict[str, Any]) -> None:
    im = torch.randn(1, 3, 224, 224)

    # 1. Random weights
    extractor = constructor(weights=None, **args).eval()
    features1 = extractor.extract(im)

    # 2. Load from file
    fname = "weights_tmp.pth"
    torch.save({"state_dict": extractor.state_dict()}, fname)
    extractor = constructor(weights=fname, **args).eval()
    features2 = extractor.extract(im)
    Path(fname).unlink()

    assert features1.ndim == 2
    assert features1.shape[-1] == extractor.feat_dim

    # 3. Test pretrained
    for weights in constructor.pretrained_models.keys():
        if (("vitl" in weights) or ("resnet101" in weights) or ("resnet152" in weights)) and SKIP_LARGE_CKPT:
            continue
        extractor = constructor.from_pretrained(weights=weights)
        extractor.extract(im)

    assert torch.allclose(features1, features2)


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
