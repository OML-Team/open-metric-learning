from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, nn

from oml.const import (
    CATEGORIES_COLUMN,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    SEQUENCE_COLUMN,
)
from oml.interfaces.datasets import IDatasetQueryGallery
from oml.utils.misc import one_hot


class IdealOneHotModel(nn.Module):
    def __init__(self, emb_dim: int, shift: int = 0):
        super(IdealOneHotModel, self).__init__()
        self.emb_dim = emb_dim
        self.shift = shift

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        embeddings = torch.stack([one_hot(label + self.shift, self.emb_dim) for label in labels])
        return embeddings


class IdealClusterEncoder(nn.Module):
    def forward(self, labels: torch.Tensor, need_noise: bool = True) -> torch.Tensor:
        embeddings = labels + need_noise * 0.01 * torch.randn_like(labels, dtype=torch.float)
        embeddings = embeddings.view((len(labels), 1))
        return embeddings


class EmbeddingsQueryGalleryDataset(IDatasetQueryGallery):
    def __init__(
        self,
        embeddings: FloatTensor,
        labels: LongTensor,
        is_query: BoolTensor,
        is_gallery: BoolTensor,
        categories: Optional[np.ndarray] = None,
        sequence: Optional[np.ndarray] = None,
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

        self.extra_data = {}
        if categories:
            self.extra_data[CATEGORIES_COLUMN] = categories

        if sequence:
            self.extra_data[SEQUENCE_COLUMN] = sequence

        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.index_key = index_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = {
            self.input_tensors_key: self._embeddings[idx],
            self.labels_key: self._labels[idx],
            self.index_key: idx,
            # todo 522: remove
            IS_QUERY_KEY: self._is_query[idx],
            IS_GALLERY_KEY: self._is_gallery[idx],
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
