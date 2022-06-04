from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset

from oml.samplers.balanced import BalanceBatchSampler  # noqa


class IDatasetWithLabels(Dataset, ABC):
    """
    Dataset with get_labels() method.
    As an example, it can be useful to initialise Sampler.
    For instance,
    >>> BalanceBatchSampler
    """

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        Args:
            item: Idx of sample

        Returns:
            Dict with the following keys:
              "input_tensors", "labels"
        """
        raise NotImplementedError

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
              'input_tensors', 'labels', 'is_query', 'is_gallery'
        """
        raise NotImplementedError()
