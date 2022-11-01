from typing import List

import torch
from torch import nn
from torchvision.ops import MLP

from oml.interfaces.models import IExtractor


class ExtractorWithLinearProjection(IExtractor):
    """
    Class-wrapper for extractors which adds additional linear layer (may be useful for classification losses).

    """

    def __init__(
        self,
        extractor: IExtractor,
        projection_feature_size: int,
    ):
        self.extractor = extractor
        self.projection = nn.Linear(self.extractor.feat_dim, projection_feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.extractor(x))

    @property
    def feat_dim(self) -> int:
        return self.projection.out_features


class ExtractorWithMLP(IExtractor):
    """
    Class-wrapper for extractors which adds additional MLP (may be useful for classification losses).

    """

    def __init__(
        self,
        extractor: IExtractor,
        mlp_features: List[int],
    ):
        self.extractor = extractor
        self.projection = MLP(self.extractor.feat_dim, mlp_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.extractor(x))

    @property
    def feat_dim(self) -> int:
        return self.projection.out_features
