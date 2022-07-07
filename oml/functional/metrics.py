import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from oml.losses.triplet import get_tri_ids_in_plain

TMetricsDict = Dict[str, Dict[int, Union[float, torch.Tensor]]]


def calc_retrieval_metrics(
    distances: torch.Tensor,
    mask_gt: torch.Tensor,
    mask_to_ignore: Optional[torch.Tensor] = None,
    top_k: Tuple[int, ...] = (1,),
    need_cmc: bool = True,
    need_precision: bool = True,
    need_map: bool = True,
    reduce: bool = True,
) -> TMetricsDict:
    """
    Function to count different retrieval metrics.
    Args:
        distances: Distance matrix shape of (query_size, gallery_size)
        mask_gt: mask_gt[i,j] indicates if for i-th query j-th gallery is the correct prediction
        mask_to_ignore: Binary matrix to indicate that some of the elements in gallery cannot be used
                     as answers and must be ignored
        top_k: Number of top examples for cumulative score counting
        need_cmc: If CMC metric is needed
        need_map: If MeanAveragePrecision metric is needed
        need_precision: If Precision metric  is needed
        reduce: if False return metrics for each query without averaging
    Returns:
        Dictionary with metrics.
    """

    if not any([need_map, need_cmc, need_precision]):
        raise ValueError("You must specify at leas 1 metric to calculate it")

    if not ((len(top_k) >= 1) and all([isinstance(x, int) and (x > 0) for x in top_k])):
        raise ValueError(f"Something is wrong with top_k: {top_k}")

    if distances.shape != mask_gt.shape:
        raise ValueError(
            f"Distances matrix has the shape of {distances.shape}, "
            f"but mask_to_ignore has the shape of {mask_gt.shape}."
        )

    if (mask_to_ignore is not None) and (mask_to_ignore.shape != distances.shape):
        raise ValueError(
            f"Distances matrix has the shape of {distances.shape}, "
            f"but mask_to_ignore has the shape of {mask_to_ignore.shape}."
        )

    query_sz, gallery_sz = distances.shape

    for k in top_k:
        if k > gallery_sz:
            warnings.warn(
                f"Your desired k={k} more than gallery_size={gallery_sz}."
                f"We'll calculate metrics with k limited by the gallery size."
            )

    if mask_to_ignore is not None:
        distances, mask_gt = apply_mask_to_ignore(distances=distances, mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)

    top_k_clipped = tuple(min(k, gallery_sz) for k in top_k)

    max_k = max(top_k_clipped)
    _, ii_top_k = torch.topk(distances, k=max_k, largest=False)
    ii_arange = torch.arange(query_sz).unsqueeze(-1).expand(query_sz, max_k)
    gt_tops = mask_gt[ii_arange, ii_top_k]

    metrics: TMetricsDict = defaultdict(dict)

    for k, k_show in zip(top_k_clipped, top_k):

        if need_cmc:
            cmc = torch.any(gt_tops[:, :k], dim=1).float()
            if reduce:
                cmc = cmc.mean()
            metrics["cmc"][k_show] = cmc

        if need_precision:
            n_gt_matrix = torch.min(mask_gt.sum(dim=1), torch.tensor(k).unsqueeze(0))
            precision = torch.sum(gt_tops[:, :k].float(), dim=1) / n_gt_matrix
            if reduce:
                precision = precision.mean()
            metrics["precision"][k_show] = precision

        if need_map:
            n_gt_matrix = torch.min(mask_gt.sum(dim=1), torch.tensor(k).unsqueeze(0))
            correct_preds = torch.cumsum(gt_tops[:, :k], dim=1)
            positions = torch.arange(1, k + 1).unsqueeze(0)
            mean_ap = torch.sum((correct_preds / positions) * gt_tops[:, :k], dim=1) / n_gt_matrix
            if reduce:
                mean_ap = mean_ap.mean()
            metrics["map"][k_show] = mean_ap

    return metrics


def apply_mask_to_ignore(
    distances: torch.Tensor, mask_gt: torch.Tensor, mask_to_ignore: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    distances[mask_to_ignore] = float("inf")
    mask_gt[mask_to_ignore] = False
    return distances, mask_gt


def calc_gt_mask(
    labels: Union[np.ndarray, torch.Tensor],
    is_query: Union[np.ndarray, torch.Tensor],
    is_gallery: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    assert all(isinstance(vector, (np.ndarray, torch.Tensor)) for vector in [labels, is_query, is_gallery])
    assert labels.ndim == is_query.ndim == is_gallery.ndim == 1
    assert len(labels) == len(is_query) == len(is_gallery)

    labels, is_query, is_gallery = map(_to_tensor, [labels, is_query, is_gallery])

    query_mask = is_query == 1
    gallery_mask = is_gallery == 1
    query_labels = labels[query_mask]
    gallery_labels = labels[gallery_mask]
    gt_mask = query_labels[..., None] == gallery_labels[None, ...]

    # TODO: add check for case if some of queries have no gallery

    return gt_mask


def calc_mask_to_ignore(
    is_query: Union[np.ndarray, torch.Tensor], is_gallery: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    assert all(isinstance(vector, (np.ndarray, torch.Tensor)) for vector in [is_query, is_gallery])
    assert is_query.ndim == is_gallery.ndim == 1
    assert len(is_query) == len(is_gallery)

    is_query, is_gallery = map(_to_tensor, [is_query, is_gallery])

    ids_query = torch.nonzero(is_query).squeeze()
    ids_gallery = torch.nonzero(is_gallery).squeeze()
    mask_to_ignore = ids_query[..., None] == ids_gallery[None, ...]

    return mask_to_ignore


def calc_distance_matrix(
    embeddings: Union[np.ndarray, torch.Tensor],
    is_query: Union[np.ndarray, torch.Tensor],
    is_gallery: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    assert all(isinstance(vector, (np.ndarray, torch.Tensor)) for vector in [embeddings, is_query, is_gallery])
    assert is_query.ndim == 1 and is_gallery.ndim == 1 and embeddings.ndim == 2
    assert embeddings.shape[0] == len(is_query) == len(is_gallery)

    embeddings, is_query, is_gallery = map(_to_tensor, [embeddings, is_query, is_gallery])

    query_mask = is_query == 1
    gallery_mask = is_gallery == 1
    query_embeddings = embeddings[query_mask]
    gallery_embeddings = embeddings[gallery_mask]

    distance_matrix = torch.cdist(query_embeddings, gallery_embeddings)

    return distance_matrix


def calculate_accuracy_on_triplets(embeddings: torch.Tensor, reduce_mean: bool = True) -> torch.Tensor:
    assert embeddings.ndim == 2
    assert embeddings.shape[0] % 3 == 0

    embeddings = embeddings.unsqueeze(1)

    anchor_ii, positive_ii, negative_ii = get_tri_ids_in_plain(n=len(embeddings))
    pos_dists = torch.cdist(x1=embeddings[anchor_ii], x2=embeddings[positive_ii]).squeeze()
    neg_dists = torch.cdist(x1=embeddings[anchor_ii], x2=embeddings[negative_ii]).squeeze()

    acc = (pos_dists < neg_dists).float()

    if reduce_mean:
        return acc.mean()
    else:
        return acc


def _to_tensor(array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise TypeError("Wrong type")
