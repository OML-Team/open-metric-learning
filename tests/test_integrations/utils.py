from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn

from oml.const import (
    CATEGORIES_COLUMN,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    LABELS_KEY,
    SEQUENCE_COLUMN,
)
from oml.functional.metrics import TMetricsDict, calc_retrieval_metrics
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
    ):
        super().__init__()
        assert len(embeddings) == len(is_query) == len(is_gallery)

        self._embeddings = embeddings
        self._is_query = is_query
        self._is_gallery = is_gallery

        self.extra_data = {}
        if categories is not None:
            assert len(categories) == len(embeddings)
            self.extra_data[CATEGORIES_COLUMN] = categories

        if sequence is not None:
            assert len(sequence) == len(embeddings)
            self.extra_data[SEQUENCE_COLUMN] = sequence

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = {
            self.input_tensors_key: self._embeddings[item],
            self.index_key: item,
        }

        for key, record in self.extra_data.items():
            if key in data:
                raise ValueError(f"<extra_data> and dataset share the same key: {key}")
            else:
                data[key] = record[item]

        return data

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

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        data[self.labels_key] = self._labels[item]
        return data

    def get_labels(self) -> np.ndarray:
        return np.array(self._labels)


def apply_mask_to_ignore(distances: Tensor, mask_gt: Tensor, mask_to_ignore: Tensor) -> Tuple[Tensor, Tensor]:
    distances[mask_to_ignore] = float("inf")
    mask_gt[mask_to_ignore] = False
    return distances, mask_gt


def calc_gt_mask(labels: Tensor, is_query: Tensor, is_gallery: Tensor) -> Tensor:
    assert labels.ndim == is_query.ndim == is_gallery.ndim == 1
    assert len(labels) == len(is_query) == len(is_gallery)

    query_mask = is_query == 1
    gallery_mask = is_gallery == 1
    query_labels = labels[query_mask]
    gallery_labels = labels[gallery_mask]
    gt_mask = query_labels[..., None] == gallery_labels[None, ...]

    return gt_mask


def calc_mask_to_ignore(
    is_query: Tensor, is_gallery: Tensor, sequence_ids: Optional[Union[Tensor, np.ndarray]] = None
) -> Tensor:
    assert is_query.ndim == is_gallery.ndim == 1
    assert len(is_query) == len(is_gallery)

    if sequence_ids is not None:
        assert sequence_ids.ndim == 1
        assert len(is_gallery) == len(sequence_ids)

    ids_query = torch.nonzero(is_query).squeeze()
    ids_gallery = torch.nonzero(is_gallery).squeeze()

    # this mask excludes duplicates of queries from the gallery if any
    mask_to_ignore = ids_query[..., None] == ids_gallery[None, ...]

    if sequence_ids is not None:
        # this mask ignores gallery samples taken from the same sequence as a given query
        mask_to_ignore_seq = sequence_ids[is_query][..., None] == sequence_ids[is_gallery][None, ...]
        mask_to_ignore = np.logical_or(mask_to_ignore, mask_to_ignore_seq)  # numpy casts tensor to numpy array
        mask_to_ignore = torch.tensor(mask_to_ignore, dtype=torch.bool)

    return mask_to_ignore


def calc_retrieval_metrics_on_full(
    distances: Tensor,
    mask_gt: Tensor,
    mask_to_ignore: Optional[Tensor] = None,
    cmc_top_k: Tuple[int, ...] = (5,),
    precision_top_k: Tuple[int, ...] = (5,),
    map_top_k: Tuple[int, ...] = (5,),
    reduce: bool = True,
) -> TMetricsDict:
    if mask_to_ignore is not None:
        distances, mask_gt = apply_mask_to_ignore(distances=distances, mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)

    max_k_arg = max([*cmc_top_k, *precision_top_k, *map_top_k])
    k = min(distances.shape[1], max_k_arg)
    _, retrieved_ids = torch.topk(distances, largest=False, k=k)

    gt_ids = [LongTensor(row.nonzero()).view(-1) for row in mask_gt]

    metrics = calc_retrieval_metrics(
        cmc_top_k=cmc_top_k,
        precision_top_k=precision_top_k,
        map_top_k=map_top_k,
        reduce=reduce,
        gt_ids=gt_ids,
        retrieved_ids=retrieved_ids,
    )
    return metrics
