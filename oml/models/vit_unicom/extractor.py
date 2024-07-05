from pathlib import Path
from typing import Optional, Union

import torch

from oml.const import CKPT_SAVE_ROOT
from oml.interfaces.models import IExtractor
from oml.models.utils import (
    remove_criterion_in_state_dict,
    remove_prefix_from_state_dict,
)
from oml.models.vit_unicom.external import vision_transformer
from oml.models.vit_unicom.external.model import load  # type: ignore
from oml.utils.misc_torch import normalise


def unicom_vitb32(using_checkpoint: bool = True) -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-B/32", using_checkpoint=using_checkpoint)  # type: ignore


def unicom_vitb16(using_checkpoint: bool = True) -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-B/16", using_checkpoint=using_checkpoint)  # type: ignore


def unicom_vitl14(using_checkpoint: bool = True) -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-L/14", using_checkpoint=using_checkpoint)  # type: ignore


def unicom_vitl14_336px(using_checkpoint: bool = True) -> vision_transformer.VisionTransformer:  # type: ignore
    return vision_transformer.build_model("ViT-L/14@336px", using_checkpoint=using_checkpoint)  # type: ignore


class ViTUnicomExtractor(IExtractor):
    constructors = {
        "vitb32_unicom": unicom_vitb32,
        "vitb16_unicom": unicom_vitb16,
        "vitl14_unicom": unicom_vitl14,
        "vitl14_336px_unicom": unicom_vitl14_336px,
    }

    pretrained_models = {
        "vitb32_unicom": {
            "download_fn": lambda using_checkpoint: load(
                "ViT-B/32", download_root=CKPT_SAVE_ROOT, using_checkpoint=using_checkpoint
            ),
            "init_args": {"arch": "vitb32_unicom", "normalise_features": True},
        },
        "vitb16_unicom": {
            "download_fn": lambda using_checkpoint: load(
                "ViT-B/16", download_root=CKPT_SAVE_ROOT, using_checkpoint=using_checkpoint
            ),
            "init_args": {"arch": "vitb16_unicom", "normalise_features": True},
        },
        "vitl14_unicom": {
            "download_fn": lambda using_checkpoint: load(
                "ViT-L/14", download_root=CKPT_SAVE_ROOT, using_checkpoint=using_checkpoint
            ),
            "init_args": {"arch": "vitl14_unicom", "normalise_features": True},
        },
        "vitl14_336px_unicom": {
            "download_fn": lambda using_checkpoint: load(
                "ViT-L/14@336px", download_root=CKPT_SAVE_ROOT, using_checkpoint=using_checkpoint
            ),
            "init_args": {"arch": "vitl14_336px_unicom", "normalise_features": True},
        },
    }

    def __init__(
        self, weights: Optional[Union[Path, str]], arch: str, normalise_features: bool, use_gradient_ckpt: bool = True
    ):
        """
        Args:
            weights: Path to weights or a special key to download pretrained checkpoint, use ``None`` to
             randomly initialize model's weights. You can check the available pretrained checkpoints
             in ``self.pretrained_models``.
            arch: Might be one of ``vitb32_unicom``, ``vitb16_unicom``, ``vitl14_unicom``, ``vitl14_336px_unicom``.
             You can check all the available options in ``self.constructors``
            normalise_features: Set ``True`` to normalise output features
            use_gradient_ckpt: Whether to use gradient checkpointing inside VisionTransformer class.
        """
        assert arch in self.constructors
        super(IExtractor, self).__init__()

        self.arch = arch
        self.normalise_features = normalise_features

        self.model = self.constructors[arch](using_checkpoint=use_gradient_ckpt)

        if weights is None:
            return
        elif weights in self.constructors:
            self.model, _ = self.pretrained_models[weights]["download_fn"](  # type: ignore
                using_checkpoint=use_gradient_ckpt
            )
        else:
            ckpt = torch.load(weights, map_location="cpu")
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            state_dict = remove_criterion_in_state_dict(state_dict)
            ckpt = remove_prefix_from_state_dict(state_dict, trial_key="norm.bias")
            self.model.load_state_dict(ckpt, strict=True)

    @property
    def feat_dim(self) -> int:
        return {"vitb32_unicom": 512, "vitb16_unicom": 768, "vitl14_unicom": 768, "vitl14_336px_unicom": 768}[self.arch]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.normalise_features:
            x = normalise(x)

        return x


__all__ = ["ViTUnicomExtractor"]
