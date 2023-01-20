from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset

from oml.const import (  # noqa
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    PAIR_1ST_KEY,
    PAIR_2ND_KEY,
)
from oml.samplers.balance import BalanceSampler  # noqa


class IDatasetWithLabels(Dataset, ABC):
    """
    This is an interface for the datasets which can provide their labels.

    """

    input_tensors_key: str = INPUT_TENSORS_KEY
    labels_key: str = LABELS_KEY
    index_key: str = INDEX_KEY

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        Args:
            item: Idx of the sample

        Returns:
             Dictionary with the following keys:

            >>> self.input_tensors_key
            >>> self.labels_key
            >>> self.index_key

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

    input_tensors_key: str = INPUT_TENSORS_KEY
    labels_key: str = LABELS_KEY
    is_query_key: str = IS_QUERY_KEY
    is_gallery_key: str = IS_GALLERY_KEY
    index_key: str = INDEX_KEY

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Args:
            item: Idx of the sample

        Returns:
             Dictionary with the following keys:

            >>> self.input_tensors_key
            >>> self.labels_key
            >>> self.is_query_key
            >>> self.is_gallery_key
            >>> self.index_key

        """
        raise NotImplementedError()


class IPairsDataset(Dataset, ABC):
    """
    This is an interface for the datasets which return pair of something.

    """

    pairs_1st_key: str = PAIR_1ST_KEY
    pairs_2nd_key: str = PAIR_2ND_KEY
    index_key: str = INDEX_KEY

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Args:
            item: Idx of the sample

        Returns:
             Dictionary with the following keys:

            >>> self.pairs_1st_key
            >>> self.pairs_2nd_key
            >>> self.index_key

        """
        raise NotImplementedError()


__all__ = ["IDatasetWithLabels", "IDatasetQueryGallery", "IPairsDataset"]
