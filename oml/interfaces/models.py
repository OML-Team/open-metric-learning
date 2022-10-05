from abc import ABC

import numpy as np
from torch import Tensor, nn


class IExtractor(nn.Module, ABC):
    """
    Models have to inherit this interface to be comparable with the rest of the library.

    """

    def extract(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def feat_dim(self) -> int:
        raise NotImplementedError()


__all__ = ["IExtractor"]
