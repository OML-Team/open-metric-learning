from typing import Any, Dict, Optional

import numpy as np
from torch import BoolTensor, FloatTensor, LongTensor

from oml.const import (
    CATEGORIES_COLUMN,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    LABELS_KEY,
    SEQUENCE_COLUMN,
)
from oml.interfaces.datasets import IDatasetQueryGallery


class EmbeddingsQueryGalleryDataset(IDatasetQueryGallery):
    """
    This dataset is currently mostly used in tests.
    """

    def __init__(
        self,
        embeddings: FloatTensor,
        labels: LongTensor,
        is_query: BoolTensor,
        is_gallery: BoolTensor,
        categories: Optional[np.ndarray] = None,
        ignoring_groups: Optional[np.ndarray] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        index_key: str = INDEX_KEY,
    ):
        super().__init__()
        assert len(embeddings) == len(labels) == len(is_query) == len(is_gallery)

        self._embeddings = embeddings
        self._labels = labels
        self._is_query = is_query
        self._is_gallery = is_gallery

        self.extra_data = {CATEGORIES_COLUMN: categories, SEQUENCE_COLUMN: ignoring_groups}

        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.index_key = index_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = {
            self.input_tensors_key: self._embeddings[idx],
            self.labels_key: self._labels[idx],
            self.index_key: idx,
        }
        return batch

    def __len__(self) -> int:
        return len(self._embeddings)

    def get_query_ids(self) -> LongTensor:
        return self._is_query.nonzero().squeeze()

    def get_gallery_ids(self) -> LongTensor:
        return self._is_gallery.nonzero().squeeze()

    def get_labels(self) -> np.ndarray:
        return np.array(self._labels)


__all__ = ["EmbeddingsQueryGalleryDataset"]
