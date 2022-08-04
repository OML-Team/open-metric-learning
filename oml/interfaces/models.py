from abc import ABC

import numpy as np
from torch import Tensor, nn


class IExtractor(nn.Module, ABC):
    def extract(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    @property
    def feat_dim(self) -> int:
        raise NotImplementedError()

    def draw_attention(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


__all__ = ["IExtractor"]
