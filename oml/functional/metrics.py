import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from oml.losses.triplet import get_tri_ids_in_plain
from oml.utils.misc import clip_max
from oml.utils.misc_torch import elementwise_dist, pairwise_dist

TMetricsDict = Dict[str, Dict[int, Union[float, torch.Tensor]]]


def calc_retrieval_metrics(
    distances: torch.Tensor,
    mask_gt: torch.Tensor,
    mask_to_ignore: Optional[torch.Tensor] = None,
    cmc_top_k: Tuple[int, ...] = (5,),
    precision_top_k: Tuple[int, ...] = (5,),
    map_top_k: Tuple[int, ...] = (5,),
    fmr_vals: Tuple[int, ...] = (1,),
    reduce: bool = True,
    check_dataset_validity: bool = False,
) -> TMetricsDict:
    """
    Function to count different retrieval metrics.

    Args:
        distances: Distance matrix with the shape of ``[query_size, gallery_size]``
        mask_gt: ``(i,j)`` element indicates if for i-th query j-th gallery is the correct prediction
        mask_to_ignore: Binary matrix to indicate that some of the elements in gallery cannot be used
                     as answers and must be ignored
        cmc_top_k: Tuple of ``k`` values to calculate ``cmc@k`` (`Cumulative Matching Characteristic`)
        precision_top_k: Tuple of  ``k`` values to calculate ``precision@k``
        map_top_k: Tuple of ``k`` values to calculate ``map@k`` (`Mean Average Precision`)
        fmr_vals: Values of ``fmr`` (measured in percents) for which we compute the corresponding ``FNMR``.
                  For example, if ``fmr_values`` is (20, 40) we will calculate ``FNMR@FMR=20`` and ``FNMR@FMR=40``
        reduce: If ``False`` return metrics for each query without averaging
        check_dataset_validity: Set ``True`` if you want to check that we have available answers in the gallery for
         each of the queries

    Returns:
        Metrics dictionary.

    """
    top_k_args = [cmc_top_k, precision_top_k, map_top_k]

    if not any(top_k_args):
        raise ValueError("You must specify arguments for at leas 1 metric to calculate it")

    if check_dataset_validity:
        validate_dataset(mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)

    for top_k_arg in top_k_args:
        if top_k_arg:
            assert all([isinstance(x, int) and (x > 0) for x in top_k_arg])

    assert all(0 <= x <= 100 for x in fmr_vals)

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

    for top_k_arg in top_k_args:
        for k in top_k_arg:
            if k > gallery_sz:
                warnings.warn(
                    f"Your desired k={k} more than gallery_size={gallery_sz}."
                    f"We'll calculate metrics with k limited by the gallery size."
                )

    if mask_to_ignore is not None:
        distances, mask_gt = apply_mask_to_ignore(distances=distances, mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)

    cmc_top_k_clipped = clip_max(cmc_top_k, gallery_sz)
    precision_top_k_clipped = clip_max(precision_top_k, gallery_sz)
    map_top_k_clipped = clip_max(map_top_k, gallery_sz)

    max_k = max([*cmc_top_k, *precision_top_k, *map_top_k])
    max_k = min(max_k, gallery_sz)

    _, ii_top_k = torch.topk(distances, k=max_k, largest=False)
    ii_arange = torch.arange(query_sz).unsqueeze(-1).expand(query_sz, max_k)
    gt_tops = mask_gt[ii_arange, ii_top_k]

    metrics: TMetricsDict = defaultdict(dict)

    for k_cmc, k_cmc_show in zip(cmc_top_k_clipped, cmc_top_k):
        cmc = torch.any(gt_tops[:, :k_cmc], dim=1).float()
        metrics["cmc"][k_cmc_show] = cmc

    for k_precision, k_precision_show in zip(precision_top_k_clipped, precision_top_k):
        n_gt_matrix = torch.min(mask_gt.sum(dim=1), torch.tensor(k_precision).unsqueeze(0))
        precision = torch.sum(gt_tops[:, :k_precision].float(), dim=1) / n_gt_matrix
        metrics["precision"][k_precision_show] = precision

    for k_map, k_map_show in zip(map_top_k_clipped, map_top_k):
        n_gt_matrix = torch.min(mask_gt.sum(dim=1), torch.tensor(k_map).unsqueeze(0))
        correct_preds = torch.cumsum(gt_tops[:, :k_map], dim=1)
        positions = torch.arange(1, k_map + 1).unsqueeze(0)
        mean_ap = torch.sum((correct_preds / positions) * gt_tops[:, :k_map], dim=1) / n_gt_matrix
        metrics["map"][k_map_show] = mean_ap

    if len(fmr_vals) > 0:
        pos_dist, neg_dist = extract_pos_neg_dists(distances, mask_gt, mask_to_ignore)
        metric_vals = calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals)
        for fmr_val, metric_val in zip(fmr_vals, metric_vals):
            metrics["fnmr@fmr"][fmr_val] = metric_val

    if reduce:
        metrics = reduce_metrics(metrics)

    return metrics


def reduce_metrics(metrics_to_reduce: TMetricsDict) -> TMetricsDict:
    output: TMetricsDict = {}

    for k, v in metrics_to_reduce.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            output[k] = v.mean()
        elif isinstance(v, (float, int)):
            output[k] = v
        else:
            output[k] = reduce_metrics(v)  # type: ignore

    return output


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

    distance_matrix = pairwise_dist(x1=query_embeddings, x2=gallery_embeddings, p=2)

    return distance_matrix


def calculate_accuracy_on_triplets(embeddings: torch.Tensor, reduce_mean: bool = True) -> torch.Tensor:
    assert embeddings.ndim == 2
    assert embeddings.shape[0] % 3 == 0

    anchor_ii, positive_ii, negative_ii = get_tri_ids_in_plain(n=len(embeddings))

    pos_dists = elementwise_dist(x1=embeddings[anchor_ii], x2=embeddings[positive_ii]).squeeze()
    neg_dists = elementwise_dist(x1=embeddings[anchor_ii], x2=embeddings[negative_ii]).squeeze()

    acc = (pos_dists < neg_dists).float()

    if reduce_mean:
        return acc.mean()
    else:
        return acc


def calc_fnmr_at_fmr(pos_dist: torch.Tensor, neg_dist: torch.Tensor, fmr_vals: Tuple[int, ...] = (1,)) -> torch.Tensor:
    """
    Function to compute False Non Match Rate (FNMR) value when False Match Rate (FMR) value
    is equal to ``fmr_val``.

    Args:
        pos_dist: distances between samples from the same class
        neg_dist: distances between samples from different classes
        fmr_vals: Values of ``fmr`` (measured in percents) for which we compute the corresponding ``FNMR``.
                  For example, if ``fmr_values`` is (20, 40) we will calculate ``FNMR@FMR=20`` and ``FNMR@FMR=40``
    Returns:
        Tensor of ``FNMR@FMR`` values.

    See:

    `Biometrics Performance`_.

    `BIOMETRIC RECOGNITION: A MODERN ERA FOR SECURITY`_.

    .. _Biometrics Performance:
        https://en.wikipedia.org/wiki/Biometrics#Performance

    .. _`BIOMETRIC RECOGNITION: A MODERN ERA FOR SECURITY`:
        https://www.researchgate.net/publication/50315614_BIOMETRIC_RECOGNITION_A_MODERN_ERA_FOR_SECURITY

    For instance, for the following distances arrays
        pos_dist: 0 0 1 1 2 2 5 5 9 9
        neg_dist: 3 3 4 4 6 6 7 7 8 8
        10 percentile of negative distances is 3
    The number of positive distances that are greater or equal than 3 is 4, therefore FNMR@FMR(10%) is 4 / 10
    """
    thresholds = torch.from_numpy(np.percentile(neg_dist.cpu().numpy(), fmr_vals)).to(pos_dist)
    fnmr_at_fmr = (pos_dist[None, :] >= thresholds[:, None]).sum(axis=1) / len(pos_dist)
    return fnmr_at_fmr


def extract_pos_neg_dists(
    distances: torch.Tensor, mask_gt: torch.Tensor, mask_to_ignore: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract distances between samples from the same class, and distances between samples
    in different classes.

    Args:
        distances: Distance matrix with the shape of ``[query_size, gallery_size]``
        mask_gt: ``(i,j)`` element indicates if for i-th query j-th gallery is the correct prediction
        mask_to_ignore: Binary matrix to indicate that some of the elements in gallery cannot be used
                     as answers and must be ignored
    Return:
        pos_dist: Tensor of distances between samples from the same class
        neg_dist: Tensor of distances between samples from diffetent classes
    """
    if mask_to_ignore is not None:
        mask_to_not_ignore = ~mask_to_ignore
        pos_dist = distances[mask_gt & mask_to_not_ignore]
        neg_dist = distances[~mask_gt & mask_to_not_ignore]
    else:
        pos_dist = distances[mask_gt]
        neg_dist = distances[~mask_gt]
    return pos_dist, neg_dist


def validate_dataset(mask_gt: torch.Tensor, mask_to_ignore: torch.Tensor) -> None:
    assert (
        (mask_gt & ~mask_to_ignore).any(1).all()
    ), "There are queries without available correct answers in the gallery!"


def _to_tensor(array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise TypeError("Wrong type")


__all__ = [
    "TMetricsDict",
    "calc_retrieval_metrics",
    "apply_mask_to_ignore",
    "calc_gt_mask",
    "calc_mask_to_ignore",
    "calc_distance_matrix",
    "calculate_accuracy_on_triplets",
    "reduce_metrics",
]
