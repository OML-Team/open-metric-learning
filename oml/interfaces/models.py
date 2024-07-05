from abc import ABC
from typing import Any, Dict

from torch import Tensor, nn


class IExtractor(nn.Module, ABC):
    """
    Models have to inherit this interface to be comparable with the rest of the library.
    """

    pretrained_models: Dict[str, Any] = {}

    def extract(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @property
    def feat_dim(self) -> int:
        """
        The only method that obligatory to implemented.
        """
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, weights: str, **kwargs) -> "IExtractor":  # type: ignore
        """
        This method allows to download a pretrained checkpoint.
        The class field ``self.pretrained_models`` is the dictionary which keeps records of all the available
        checkpoints in the format, depending on implementation of a particular child of ``IExtractor``.
        As a user, you don't need to worry about implementing this method.

        Args:
            weights: A unique identifier (key) of a pretrained model information stored in
              a class field ``self.pretrained_models``.

        Returns: An instance of ``IExtractor``

        """
        if weights not in cls.pretrained_models:
            raise KeyError(
                f"There is no pretrained model {weights}. The existing ones are {list(cls.pretrained_models.keys())}."
            )

        extractor = cls(weights=weights, **cls.pretrained_models[weights]["init_args"], **kwargs)  # type: ignore
        return extractor


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
    """
    A model of this type takes two inputs, for example, two embeddings or two images.

    """

    def forward(self, x1: Any, x2: Any) -> Tensor:
        """

        Args:
            x1: The first input.
            x2: The second input.

        """
        raise NotImplementedError()

    def predict(self, x1: Any, x2: Any) -> Tensor:
        """
        While ``self.forward()`` is called during training, this method is called during
        inference or validation time. For example, it allows application of some activation,
        which was a part of a loss function during the training.

        Args:
            x1: The first input.
            x2: The second input.

        """
        raise NotImplementedError()


__all__ = ["IExtractor", "IFreezable", "IPairwiseModel"]
