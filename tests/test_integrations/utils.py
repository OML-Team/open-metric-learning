from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, nn

from oml.const import (
    CATEGORIES_KEY,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    SEQUENCE_KEY,
)
from oml.interfaces.datasets import IQueryGalleryDataset, IQueryGalleryLabeledDataset
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


class EmbeddingsQueryGalleryDataset(IQueryGalleryDataset):
    def __init__(
        self,
        embeddings: FloatTensor,
        is_query: BoolTensor,
        is_gallery: BoolTensor,
        categories: Optional[np.ndarray] = None,
        sequence: Optional[np.ndarray] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
        # todo 522: remove keys later
        categories_key: str = CATEGORIES_KEY,
        sequence_key: str = SEQUENCE_KEY,
    ):
        super().__init__()
        assert len(embeddings) == len(is_query) == len(is_gallery)

        self._embeddings = embeddings
        self._is_query = is_query
        self._is_gallery = is_gallery

        # todo 522: remove keys
        self.categories_key = categories_key
        self.sequence_key = sequence_key

        self.extra_data = {}
        if categories is not None:
            self.extra_data[self.categories_key] = categories

        if sequence is not None:
            self.extra_data[self.sequence_key] = sequence

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = {
            self.input_tensors_key: self._embeddings[idx],
            self.index_key: idx,
            # todo 522: remove
            IS_QUERY_KEY: self._is_query[idx],
            IS_GALLERY_KEY: self._is_gallery[idx],
        }

        # todo 522: avoid passing extra data as keys
        if self.extra_data:
            for key, record in self.extra_data.items():
                if key in batch:
                    raise ValueError(f"<extra_data> and dataset share the same key: {key}")
                else:
                    batch[key] = record[idx]

        return batch

    def __len__(self) -> int:
        return len(self._embeddings)

    def get_query_ids(self) -> LongTensor:
        return self._is_query.nonzero().squeeze()

    def get_gallery_ids(self) -> LongTensor:
        return self._is_gallery.nonzero().squeeze()


class EmbeddingsQueryGalleryLabeledDataset(EmbeddingsQueryGalleryDataset, IQueryGalleryLabeledDataset):
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
        super().__init__(
            embeddings=embeddings,
            is_query=is_query,
            is_gallery=is_gallery,
            categories=categories,
            sequence=sequence,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
        )

        assert len(embeddings) == len(labels)

        self._labels = labels
        self.labels_key = labels_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item[self.labels_key] = self._labels[idx]
        return item

    def get_labels(self) -> np.ndarray:
        return np.array(self._labels)
