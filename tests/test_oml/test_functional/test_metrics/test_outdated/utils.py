from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import LongTensor, Tensor

from oml.functional.metrics import TMetricsDict, calc_retrieval_metrics


def calc_retrieval_metrics_on_matrices(
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
