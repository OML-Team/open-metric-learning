from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
from torch import LongTensor
from torch.utils.data import Dataset

from oml.const import INPUT_TENSORS_KEY_1, INPUT_TENSORS_KEY_2, LABELS_KEY, TColor


class IIndexedDataset(Dataset, ABC):
    index_key: str

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        Args:
            item: Idx of the sample

        Returns:
            Dictionary having the following key:
            ``self.index_key: int = item``

        """
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


class IBaseDataset(IIndexedDataset, ABC):
    input_tensors_key: str
    extra_data: Dict[str, Any]  # container for storing extra records having the same size as the dataset

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        Args:
            item: Idx of the sample

        Returns:
            Dictionary including the following keys:
            ``self.input_tensors_key``
            ``self.index_key: int = item``

        """
        raise NotImplementedError()


class ILabeledDataset(IBaseDataset, ABC):
    """
    This is an interface for the datasets which provide labels of containing items.

    """

    labels_key: str = LABELS_KEY

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """

        Args:
            item: Idx of the sample

        Returns:
             Dictionary including the following keys:

            ``self.input_tensors_key``
            ``self.index_key: int = item``
            ``self.labels_key``

        """
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        raise NotImplementedError()

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        """
        Returns:
            Mapping from label to category if known.

        """
        raise NotImplementedError()


class IQueryGalleryDataset(IBaseDataset, ABC):
    """
    This is an interface for the datasets which hold the information on how to split
    the data into the query and gallery. The query and gallery ids may overlap.
    It doesn't need the ground truth labels, so it can be used for prediction on not annotated data.

    """

    @abstractmethod
    def get_query_ids(self) -> LongTensor:
        raise NotImplementedError()

    @abstractmethod
    def get_gallery_ids(self) -> LongTensor:
        raise NotImplementedError()


class IQueryGalleryLabeledDataset(IQueryGalleryDataset, ILabeledDataset, ABC):
    """
    This interface is similar to `IQueryGalleryDataset`, but there are ground truth labels.
    """


class IPairDataset(IIndexedDataset):
    """
    This is an interface for the datasets which return pair of something.

    """

    input_tensors_key_1: str = INPUT_TENSORS_KEY_1
    input_tensors_key_2: str = INPUT_TENSORS_KEY_2

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Args:
            item: Idx of the sample

        Returns:
             Dictionary with the following keys:

            ``self.input_tensors_key_1``
            ``self.input_tensors_key_2``
            ``self.index_key``

        """
        raise NotImplementedError()


class IVisualizableDataset(Dataset, ABC):
    """
    Base class for the datasets which know how to visualise their items.
    """

    @abstractmethod
    def visualize(self, item: int, color: TColor) -> np.ndarray:
        raise NotImplementedError()


__all__ = [
    "IIndexedDataset",
    "IBaseDataset",
    "ILabeledDataset",
    "IQueryGalleryLabeledDataset",
    "IQueryGalleryDataset",
    "IPairDataset",
    "IVisualizableDataset",
]
