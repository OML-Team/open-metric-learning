from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch import nn
from torchvision.ops import MLP

from oml.const import STORAGE_CKPTS
from oml.interfaces.models import IExtractor, IFreezable
from oml.models.utils import remove_prefix_from_state_dict
from oml.models.vit.vit import ViTExtractor
from oml.utils.io import download_checkpoint


def get_vit_and_mlp(
    arch_vit: str,
    normalise_features_vit: bool,
    mlp_features: List[int],
    use_multi_scale_vit: bool = False,
) -> nn.Module:
    """
    Function for creation of ViT model and MLP projection.
    """
    vit = ViTExtractor(
        weights=None,
        arch=arch_vit,
        normalise_features=normalise_features_vit,
        use_multi_scale=use_multi_scale_vit,
    )
    mlp = MLP(vit.feat_dim, mlp_features)
    return vit, mlp


class ExtractorWithMLP(IExtractor, IFreezable):
    """
    Class-wrapper for extractors which adds additional MLP (may be useful for classification losses).

    """

    constructors = {
        "vits16_224_mlp_384": partial(
            get_vit_and_mlp,
            arch_vit="vits16",
            normalise_features_vit=False,
            use_multi_scale_vit=False,
            mlp_features=[384],
        )
    }

    pretrained_models = {
        "vits16_224_mlp_384_inshop": (
            f"{STORAGE_CKPTS}/inshop/vits16_224_mlp_384_inshop.ckpt",
            "35244966",
            "vits16_224_mlp_384_inshop.ckpt",
        )
    }

    def __init__(
        self,
        extractor: IExtractor,
        mlp_features: List[int],
        weights: Optional[Union[str, Path]] = None,
        strict_load: bool = True,
        train_backbone: bool = False,
    ):
        """
        Args:
            extractor: Instance of IExtractor (e.g. ViTExtractor)
            mlp_features: Sizes of projection layers
            weights: Path to weights file or ``None`` for random initialization
            strict_load: Whether to use ``self.load_state_dict`` with strict argument
        """
        IExtractor.__init__(self)
        self.train_backbone = train_backbone
        self.extractor = extractor
        self.projection = MLP(self.extractor.feat_dim, mlp_features)
        if weights:
            if weights in self.pretrained_models:
                url_or_fid, hash_md5, fname = self.pretrained_models[weights]  # type: ignore
                weights = download_checkpoint(url_or_fid=url_or_fid, hash_md5=hash_md5, fname=fname)
            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remove_prefix_from_state_dict(loaded, trial_key="extractor.")
            self.load_state_dict(loaded, strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_backbone:
            features = self.extractor(x)
        else:
            with torch.no_grad():
                features = self.extractor(x)
        return self.projection(features)

    @property
    def feat_dim(self) -> int:
        return self.projection.out_features

    def freeze(self) -> None:
        self.train_backbone = False

    def unfreeze(self) -> None:
        self.train_backbone = True


__all__ = ["ExtractorWithMLP", "get_vit_and_mlp"]
