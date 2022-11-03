from pathlib import Path
from typing import List, Optional, Union

import torch
from torch import nn
from torchvision.ops import MLP

from oml.interfaces.models import IExtractor
from oml.models.resnet import ResnetExtractor
from oml.models.vit.clip import ViTCLIPExtractor
from oml.models.vit.vit import ViTExtractor


class ExtractorWithMLP(IExtractor):
    """
    Class-wrapper for extractors which adds additional MLP (may be useful for classification losses).

    """

    def __init__(
        self,
        extractor: IExtractor,
        mlp_features: List[int],
        weights: Optional[Union[str, Path]] = None,
        strict_load: bool = True,
    ):
        """
        Args:
            extractor: Instance of IExtractor (e.g. ViTExtractor)
            mlp_features: Sizes of projection layers
            weights: Path to weights file or None for random initialization
            strict_load: Whether to use ``self.load_state_dict`` with strict argument
        """
        super().__init__()
        self.extractor = extractor
        self.projection = MLP(self.extractor.feat_dim, mlp_features)
        if weights:
            loaded = torch.load(weights)
            self.load_state_dict(loaded.get("state_dict", loaded), strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.extractor(x))

    @property
    def feat_dim(self) -> int:
        return self.projection.out_features


def vit_with_mlp(
    weights_vit: Optional[Union[Path, str]],
    arch_vit: str,
    normalise_features_vit: bool,
    mlp_features: List[int],
    use_multi_scale_vit: bool = False,
    strict_load_vit: bool = True,
    weights_mlp: Optional[Union[str, Path]] = None,
    strict_load_mlp: bool = True,
) -> nn.Module:
    """
    Function for creation of ViT model with additional MLP projection.
    """
    vit = ViTExtractor(
        weights=weights_vit,
        arch=arch_vit,
        normalise_features=normalise_features_vit,
        use_multi_scale=use_multi_scale_vit,
        strict_load=strict_load_vit,
    )
    return ExtractorWithMLP(extractor=vit, mlp_features=mlp_features, weights=weights_mlp, strict_load=strict_load_mlp)


def vit_clip_with_mlp(
    weights_vit_clip: Optional[str],
    arch_vit_clip: str,
    normalise_features_vit_clip: bool,
    mlp_features: List[int],
    strict_load_vit_clip: bool = True,
    weights_mlp: Optional[Union[str, Path]] = None,
    strict_load_mlp: bool = True,
) -> nn.Module:
    """
    Function for creation of ViT-CLIP model with additional MLP projection.
    """
    vit_clip = ViTCLIPExtractor(
        weights=weights_vit_clip,
        arch=arch_vit_clip,
        normalise_features=normalise_features_vit_clip,
        strict_load=strict_load_vit_clip,
    )
    return ExtractorWithMLP(
        extractor=vit_clip, mlp_features=mlp_features, weights=weights_mlp, strict_load=strict_load_mlp
    )


def resnet_with_mlp(
    weights_resnet: Optional[Union[Path, str]],
    arch_resnet: str,
    normalise_features_resnet: bool,
    mlp_features: List[int],
    remove_fc_resnet: bool = False,
    gem_p_resnet: Optional[float] = None,
    strict_load_resnet: bool = True,
    weights_mlp: Optional[Union[str, Path]] = None,
    strict_load_mlp: bool = True,
) -> nn.Module:
    """
    Function for creation of ResNet model with additional MLP projection.
    """
    resnet = ResnetExtractor(
        weights=weights_resnet,
        arch=arch_resnet,
        normalise_features=normalise_features_resnet,
        gem_p=gem_p_resnet,
        remove_fc=remove_fc_resnet,
        strict_load=strict_load_resnet,
    )
    return ExtractorWithMLP(
        extractor=resnet, mlp_features=mlp_features, weights=weights_mlp, strict_load=strict_load_mlp
    )
