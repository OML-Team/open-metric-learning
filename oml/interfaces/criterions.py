import re
from typing import List, Union

from torch import Tensor
from torch.nn import Module


class ITripletLossWithMiner(Module):
    def forward(self, features: Tensor, labels: Union[Tensor, List[int]]) -> Tensor:
        """
        Args:
            features: Features with shape [batch_size, features_dim]
            labels: Labels of samples which will be used to form triplets

        Returns:
            Loss value

        """
        raise NotImplementedError()


class ICriterion(Module):
    @property
    def cirt_name(self):
        # class name to snake case
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower().replace("_loss", "")
