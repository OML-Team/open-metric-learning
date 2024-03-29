from pathlib import Path
from typing import Optional, Union

import torch

from oml.interfaces.models import IExtractor
from oml.models.utils import (
    remove_criterion_in_state_dict,
    remove_prefix_from_state_dict,
)
from oml.models.vit_dinov2.external.hubconf import (  # type: ignore
    dinov2_vitb14,
    dinov2_vitb14_reg,
    dinov2_vitl14,
    dinov2_vitl14_reg,
    dinov2_vits14,
    dinov2_vits14_reg,
)
from oml.utils.io import download_checkpoint_one_of
from oml.utils.misc_torch import normalise

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


class ViTExtractor_v2(IExtractor):
    """
    The base class for the extractors that follow VisualTransformer architecture.

    """

    constructors = {
        "vits14": dinov2_vits14,
        "vitb14": dinov2_vitb14,
        "vitl14": dinov2_vitl14,
        "vits14_reg": dinov2_vits14_reg,
        "vitb14_reg": dinov2_vitb14_reg,
        "vitl14_reg": dinov2_vitl14_reg,
    }

    pretrained_models = {
        "vits14_dinov2": {
            "url": f"{_DINOV2_BASE_URL}/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "hash": "2e405c",
            "fname": "dinov2_vits14.ckpt",
            "init_args": {"arch": "vits14", "normalise_features": False},
        },
        "vits14_reg_dinov2": {
            "url": f"{_DINOV2_BASE_URL}/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
            "hash": "2a50c5",
            "fname": "dinov2_vits14_reg4.ckpt",
            "init_args": {"arch": "vits14_reg", "normalise_features": False},
        },
        "vitb14_dinov2": {
            "url": f"{_DINOV2_BASE_URL}/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
            "hash": "8635e7",
            "fname": "dinov2_vitb14.ckpt",
            "init_args": {"arch": "vitb14", "normalise_features": False},
        },
        "vitb14_reg_dinov2": {
            "url": f"{_DINOV2_BASE_URL}/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
            "hash": "13d13c",
            "fname": "dinov2_vitb14_reg4.ckpt",
            "init_args": {"arch": "vitb14_reg", "normalise_features": False},
        },
        "vitl14_dinov2": {
            "url": f"{_DINOV2_BASE_URL}/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            "hash": "19a02c",
            "fname": "dinov2_vitl14.ckpt",
            "init_args": {"arch": "vitl14", "normalise_features": False},
        },
        "vitl14_reg_dinov2": {
            "url": f"{_DINOV2_BASE_URL}/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
            "hash": "8b6364",
            "fname": "dinov2_vitl14_reg4.ckpt",
            "init_args": {"arch": "vitl14_reg", "normalise_features": False},
        },
    }

    def __init__(
        self,
        arch: str,
        normalise_features: bool,
        weights: Optional[Union[Path, str]] = None,
    ):
        assert arch in self.constructors
        super().__init__()

        self.normalise_features = normalise_features
        self.arch = arch

        factory_fun = self.constructors[self.arch]

        self.model = factory_fun(pretrained=False)
        if weights is None:
            return

        if weights in self.pretrained_models:
            pretrained = self.pretrained_models[weights]  # type: ignore
            weights = download_checkpoint_one_of(
                url_or_fid_list=pretrained["url"],  # type: ignore
                hash_md5=pretrained["hash"],  # type: ignore
                fname=pretrained["fname"],  # type: ignore
            )
        ckpt = torch.load(weights, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        state_dict = remove_criterion_in_state_dict(state_dict)
        ckpt = remove_prefix_from_state_dict(state_dict, trial_key="norm.bias")
        self.model.load_state_dict(ckpt, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.normalise_features:
            x = normalise(x)

        return x

    @property
    def feat_dim(self) -> int:
        return len(self.model.norm.bias)


__all__ = ["ViTExtractor_v2"]
