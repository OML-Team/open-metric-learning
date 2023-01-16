from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from oml.const import (  # noqa
    INPUT_TENSORS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    PAIR_1ST_KEY,
    PAIR_2ND_KEY
)
from oml.samplers.balance import BalanceSampler  # noqa
from torch.utils.data import Dataset


class IDatasetWithLabels(Dataset, ABC):
    """
    This is an interface for the datasets which can provide their labels.

    """

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        Args:
            item: Idx of the sample

        Returns:
             Dictionary with the following keys:

            >>> INPUT_TENSORS_KEY
            >>> LABELS_KEY

        """
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        raise NotImplementedError()


class IDatasetQueryGallery(Dataset, ABC):
    """
    This is an interface for the datasets which can provide the information on how to split
    the validation set into the two parts: query and gallery.

    """

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Args:
            item: Idx of the sample

        Returns:
             Dictionary with the following keys:

            >>> INPUT_TENSORS_KEY
            >>> LABELS_KEY
            >>> IS_QUERY_KEY
            >>> IS_GALLERY_KEY

        """
        raise NotImplementedError()


class IPairsDataset(Dataset, ABC):
    """
    This is an interface for the datasets which return pair of something.

    """

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Args:
            item: Idx of the sample

        Returns:
             Dictionary with the following keys:

            >>> PAIR_1ST_KEY
            >>> PAIR_2ND_KEY

        """
        raise NotImplementedError()


__all__ = ["IDatasetWithLabels", "IDatasetQueryGallery", "IPairsDataset"]
