from pathlib import Path
from typing import Optional, Union

import torch

from oml.const import CKPT_SAVE_ROOT
from oml.interfaces.models import IExtractor
from oml.models.utils import remove_prefix_from_state_dict
from oml.models.vit_unicom.external import vision_transformer
from oml.models.vit_unicom.external.model import load  # type: ignore
from oml.utils.misc_torch import normalise


def unicom_vitb32() -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-B/32")  # type: ignore


def unicom_vitb16() -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-B/16")  # type: ignore


def unicom_vitl14() -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-L/14")  # type: ignore


def unicom_vitl14_336px() -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-L/14@336px")  # type: ignore


class ViTUnicomExtractor(IExtractor):
    constructors = {
        "vitb32_unicom": unicom_vitb32,
        "vitb16_unicom": unicom_vitb16,
        "vitl14_unicom": unicom_vitl14,
        "vitl14_336px_unicom": unicom_vitl14_336px,
    }

    pretrained_models = {
        "vitb32_unicom": {
            "download_fn": lambda: load("ViT-B/32", download_root=CKPT_SAVE_ROOT),
            "fname": "FP16-ViT-B-32.pt",
            "init_args": {"arch": "vitb32_unicom", "normalise_features": True},
        },
        "vitb16_unicom": {
            "download_fn": lambda: load("ViT-B/16", download_root=CKPT_SAVE_ROOT),
            "fname": "FP16-ViT-B-16.pt",
            "init_args": {"arch": "vitb16_unicom", "normalise_features": True},
        },
        "vitl14_unicom": {
            "download_fn": lambda: load("ViT-L/14", download_root=CKPT_SAVE_ROOT),
            "fname": "FP16-ViT-L-14.pt",
            "init_args": {"arch": "vitl14_unicom", "normalise_features": True},
        },
        "vitl14_336px_unicom": {
            "download_fn": lambda: load("ViT-L/14@336px", download_root=CKPT_SAVE_ROOT),
            "fname": "FP16-ViT-L-14-336px.pt",
            "init_args": {"arch": "vitl14_336px_unicom", "normalise_features": True},
        },
    }

    def __init__(self, weights: Optional[Union[Path, str]], arch: str, normalise_features: bool):
        assert arch in self.constructors
        super(IExtractor, self).__init__()

        self.arch = arch
        self.normalise_features = normalise_features

        self.model = self.constructors[arch]()

        if weights is None:
            return
        elif weights in self.constructors:
            self.model, _ = self.pretrained_models[weights]["download_fn"]()  # type: ignore
        else:
            ckpt = torch.load(weights, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            ckpt = remove_prefix_from_state_dict(state_dict, trial_key="norm.bias")
            self.model.load_state_dict(ckpt, strict=True)

    @property
    def feat_dim(self) -> int:
        return {"vitb32_unicom": 512, "vitb16_unicom": 768, "vitl14_unicom": 768, "vitl14_336px_unicom": 768}[self.arch]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.normalise_features:
            x = normalise(x)

        print(x.shape)

        return x


__all__ = ["ViTUnicomExtractor"]
