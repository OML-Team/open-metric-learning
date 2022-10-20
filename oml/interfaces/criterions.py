from typing import List, Union

from torch import Tensor
from torch.nn import Module


class ITripletLossWithMiner(Module):
    """
    Base class for TripletLoss combined with Miner.

    """

    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tensor:
        """
        Args:
            features: Features with the shape ``[batch_size, features_dim]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Loss value

        """
        raise NotImplementedError()


__all__ = ["ITripletLossWithMiner"]
