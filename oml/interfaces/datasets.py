from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from torch import LongTensor
from torch.utils.data import Dataset

from oml.const import INDEX_KEY, LABELS_KEY, PAIR_1ST_KEY, PAIR_2ND_KEY, TColor


class IBaseDataset(Dataset):
    input_tensors_key: str
    index_key: str
    extra_data: Dict[str, Any]

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


class IDatasetWithLabels(IBaseDataset, ABC):
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

            ``self.labels_key``

        """
        raise NotImplementedError()

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        raise NotImplementedError()


class IDatasetQueryGalleryPrediction(IBaseDataset, ABC):
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


class IDatasetQueryGallery(IDatasetQueryGalleryPrediction, IDatasetWithLabels, ABC):
    """
    This class is similar to "IDatasetQueryGalleryPrediction", but we also have ground truth labels.
    """


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

            ``self.pairs_1st_key``
            ``self.pairs_2nd_key``
            ``self.index_key``

        """
        raise NotImplementedError()


class IVisualizableDataset(Dataset, ABC):
    """
    Base class for the datasets which know how to visualise their items.
    """

    @abstractmethod
    def visualize(self, idx: int, color: TColor) -> np.ndarray:
        raise NotImplementedError()


__all__ = [
    "IBaseDataset",
    "IDatasetWithLabels",
    "IDatasetQueryGallery",
    "IDatasetQueryGalleryPrediction",
    "IPairsDataset",
    "IVisualizableDataset",
]
