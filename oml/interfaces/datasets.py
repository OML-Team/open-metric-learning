from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset

from oml.const import INPUT_TENSORS_KEY, IS_GALLERY_KEY, IS_QUERY_KEY, LABELS_KEY
from oml.samplers.balance import BalanceSampler  # noqa


class IDatasetWithLabels(Dataset, ABC):
    """
    Dataset with get_labels() method.
    As an example, it can be useful to initialise Sampler.
    For instance,
    >>> BalanceSampler
    """

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        Args:
            item: Idx of sample

        Returns:
            Dict with the following keys:
            >>> INPUT_TENSORS_KEY
            >>> LABELS_KEY
        """
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """
        Raises:
            NotImplementedError: You should implement it

        """
        raise NotImplementedError()


class IDatasetQueryGallery(Dataset, ABC):
    """
    QueryGalleryDataset
    """

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Args:
            item: Idx of sample

        Returns:
            Dict with the following keys:
            >>> INPUT_TENSORS_KEY
            >>> LABELS_KEY
            >>> IS_QUERY_KEY
            >>> IS_GALLERY_KEY

        """
        raise NotImplementedError()


__all__ = ["IDatasetWithLabels", "IDatasetQueryGallery"]
