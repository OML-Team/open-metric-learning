from abc import ABC

from torch import Tensor, nn
from typing import Any


class IExtractor(nn.Module, ABC):
    """
    Models have to inherit this interface to be comparable with the rest of the library.
    """

    def extract(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def feat_dim(self) -> int:
        raise NotImplementedError()


class IFreezable(ABC):
    """
    Models which can freeze and unfreeze their parts.
    """

    def freeze(self) -> None:
        """
        Function for freezing. You can use it to partially freeze a model.
        """
        raise NotImplementedError()

    def unfreeze(self) -> None:
        """
        Function for unfreezing. You can use it to unfreeze a model.
        """
        raise NotImplementedError()


class IPairwiseModel(nn.Module):

    def forward(self, x1: Any, x2: Any) -> Tensor:
        raise NotImplementedError()


__all__ = ["IExtractor", "IFreezable", "IPairwiseModel"]
