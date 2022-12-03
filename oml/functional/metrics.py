import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
        mask_gt: ``(i,j)`` element indicates if for ``i``-th query ``j``-th gallery is the correct prediction
        mask_to_ignore: Binary matrix to indicate that some of the elements in gallery cannot be used
                     as answers and must be ignored
        cmc_top_k: Values of ``k`` values to calculate ``cmc@k`` (`Cumulative Matching Characteristic`)
        precision_top_k: Values of  ``k`` values to calculate ``precision@k``
        map_top_k: Values of ``k`` values to calculate ``map@k`` (`Mean Average Precision`)
        fmr_vals: Values of ``fmr`` values (measured in percents) to calculate ``fnmr@fmr`` (False Non Match Rate
                  at the given False Match Rate).
                  For example, if ``fmr_values`` is (20, 40) we will calculate ``fnmr@fmr=20`` and ``fnmr@fmr=40``
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
    n_gt = mask_gt.sum(dim=1)

    metrics: TMetricsDict = defaultdict(dict)

    if cmc_top_k:
        cmc = calc_cmc(gt_tops, cmc_top_k_clipped)
        metrics["cmc"] = dict(zip(cmc_top_k, cmc))

    if precision_top_k:
        precision = calc_precision(gt_tops, n_gt, precision_top_k_clipped)
        metrics["precision"] = dict(zip(precision_top_k, precision))

    if map_top_k:
        map = calc_map(gt_tops, n_gt, map_top_k_clipped)
        metrics["map"] = dict(zip(map_top_k, map))

    if fmr_vals:
        pos_dist, neg_dist = extract_pos_neg_dists(distances, mask_gt, mask_to_ignore)
        fnmr_at_fmr = calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals)
        metrics["fnmr@fmr"] = dict(zip(fmr_vals, fnmr_at_fmr))

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


def calc_cmc(gt_tops: torch.Tensor, top_k: Tuple[int, ...]) -> List[torch.Tensor]:
    """
    Function to compute Cumulative Matching Characteristics (CMC) at cutoffs ``top_k``.

    ``cmc@k`` for a given query equals to 1 if there is at least 1 instance related to this query in top ``k``
    gallery instances sorted by distances to the query, and 0 otherwise.
    The final ``cmc@k`` could be obtained by averaging the results calculated for each query.

    Args:
        gt_tops: Matrix where the ``(i, j)`` element indicates if ``j``-th gallery sample is related to
                 ``i``-th query or not. Obtained from the full ground truth matrix by taking ``max(top_k)`` elements
                 with the smallest distances to the corresponding queries.
        top_k: Values of ``k`` values to calculate ``cmc@k`` for.

    Returns:
        List of ``cmc@k`` tensors.

    .. math::
        \\textrm{cmc}@k = \\begin{cases}
        1, & \\textrm{if top-}k \\textrm{ ranked gallery samples include an output relevant to the query}, \\\\
        0, & \\textrm{otherwise}.
        \\end{cases}

    Example:
        >>> gt_tops = torch.tensor([
        ...                         [1, 0, 0],
        ...                         [0, 1, 1],
        ...                         [0, 0, 1]
        ... ], dtype=torch.bool)
        >>> calc_cmc(gt_tops, top_k=(1, 2, 3))
        [tensor([1., 0., 0.]), tensor([1., 1., 0.]), tensor([1., 1., 1.])]
    """
    _check_if_nonempty_integers_and_positive(top_k, "top_k")
    top_k = _clip_max_with_warning(top_k, gt_tops.shape[1])
    cmc = []
    for k in top_k:
        cmc.append(torch.any(gt_tops[:, :k], dim=1).float())
    return cmc


def calc_precision(gt_tops: torch.Tensor, n_gt: torch.Tensor, top_k: Tuple[int, ...]) -> List[torch.Tensor]:
    """
    Function to compute Precision at cutoffs ``top_k``.

    ``precision@k`` for a given query is a fraction of the relevant gallery instances among the top ``k`` instances
    sorted by distances from the query to the gallery.
    The final ``precision@k`` could be obtained by averaging the results calculated for each query.

    Args:
        gt_tops: Matrix where the ``(i, j)`` element indicates if ``j``-th gallery sample is related to
                 ``i``-th query or not. Obtained from the full ground truth matrix by taking ``max(top_k)`` elements
                 with the smallest distances to the corresponding queries.
        n_gt: Array where the ``i``-th element is the total number of elements in the gallery relevant
              to ``i``-th query.
        top_k: Values of ``k`` values to calculate ``precision@k`` for.

    Returns:
        List of ``precision@k`` tensors.

    Given a list of ground truth top :math:`k` closest elements from the gallery to a given query
    :math:`g = [g_1, \\ldots, g_k]` (:math:`g_i` is 1 if :math:`i`-th element from the gallery is relevant
    to the query), and the total number of relevant elements from the gallery :math:`n`,
    the :math:`\\textrm{precision}@k` for the query is defined as

    .. math::
        \\textrm{precision}@k = \\frac{1}{\\min{\\left(k, n\\right)}}\\sum\\limits_{i = 1}^k g_i

    It's worth mentioning that OML version of :math:`\\textrm{precision}@k` differs from the commonly used by
    the denominator of the fraction. The OML version takes into account the total amount of relevant elements in
    the gallery, so it will not penalize the ideal model if :math:`n < k`.

    For instance, let :math:`n = 3` and :math:`g = [1, 1, 1, 0, 0]`. Then by using the common definition of
    :math:`\\textrm{precision}@k` we get

    .. math::
        \\begin{align}
            \\textrm{precision}@1 &= \\frac{1}{1}, \\textrm{precision}@2 = \\frac{2}{2},
            \\textrm{precision}@3 = \\frac{3}{3}, \\\\
            \\textrm{precision}@4 &= \\frac{3}{4}, \\textrm{precision}@5 = \\frac{3}{5},
            \\textrm{precision}@6 = \\frac{3}{6} \\\\
        \\end{align}

    But with OML definition of :math:`\\textrm{precision}@k` we get


    .. math::
        \\begin{align}
            \\textrm{precision}@1 &= \\frac{1}{1}, \\textrm{precision}@2 = \\frac{2}{2},
            \\textrm{precision}@3 = \\frac{3}{3} \\\\
            \\textrm{precision}@4 &= \\frac{3}{3}, \\textrm{precision}@5 = \\frac{3}{3},
            \\textrm{precision}@6 = \\frac{3}{3} \\\\
        \\end{align}

    See:
        `Evaluation measures (information retrieval). Precision@k`_


        .. _`Evaluation measures (information retrieval). Precision@k`:
            https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_k

    Example:
        >>> gt_tops = torch.tensor([
        ...                         [1, 0, 0],
        ...                         [0, 1, 1],
        ...                         [0, 0, 1]
        ... ], dtype=torch.bool)
        >>> n_gt = torch.tensor([2, 3, 5])
        >>> calc_precision(gt_tops, n_gt, top_k=(1, 2, 3))
        [tensor([1., 0., 0.]), tensor([0.5000, 0.5000, 0.0000]), tensor([0.5000, 0.6667, 0.3333])]
    """
    _check_if_nonempty_integers_and_positive(top_k, "top_k")
    top_k = _clip_max_with_warning(top_k, gt_tops.shape[1])
    precision = []
    correct_preds = torch.cumsum(gt_tops.float(), dim=1)
    for k in top_k:
        _n_gt = torch.min(n_gt, torch.tensor(k).unsqueeze(0))
        precision.append(correct_preds[:, k - 1] / _n_gt)
    return precision


def calc_map(gt_tops: torch.Tensor, n_gt: torch.Tensor, top_k: Tuple[int, ...]) -> List[torch.Tensor]:
    """
    Function to compute Mean Average Precision (MAP) at cutoffs ``top_k``.

    ``map@k`` for a given query is the average value of the ``precision`` considered as a function of the ``recall``.
    The final ``map@k`` could be obtained by averaging the results calculated for each query.

    Args:
        gt_tops: Matrix where the ``(i, j)`` element indicates if ``j``-th gallery sample is related to
                 ``i``-th query or not. Obtained from the full ground truth matrix by taking ``max(top_k)`` elements
                 with the smallest distances to the corresponding queries.
        n_gt: Array where the ``i``-th element is the total number of elements in the gallery relevant
              to ``i``-th query.
        top_k: Values of ``k`` values to calculate ``map@k`` for.

    Returns:
        List of ``map@k`` tensors.

    Given a list of ground truth top :math:`k` closest elements from the gallery to a given query
    :math:`[g_1, \\ldots, g_k]` (:math:`g_i` is 1 if :math:`i`-th element from the gallery is relevant to the query),
    and the total number of relevant elements from the gallery :math:`n`, the :math:`\\textrm{map}@k`
    for the query is defined as

    .. math::
        \\begin{split}\\textrm{map}@k &=
        \\frac{1}{\\min{\\left(k, n\\right)}}\\sum\\limits_{i = 1}^k
        \\frac{\\textrm{# of relevant elements among top } i\\textrm{ elements}}{i} \\times \\textrm{rel}(i) = \\\\
        & = \\frac{1}{\\min{\\left(k, n\\right)}}\\sum\\limits_{i = 1}^k
        \\frac{\\sum\\limits_{j = 1}^{i}g_i}{i} \\times \\textrm{rel}(i)
        \\end{split}

    where :math:`\\textrm{rel}(i)` is 1 if :math:`i`-th element from the top :math:`i` closest
    elements from the gallery to the query is relevant to the query, and 0 otherwise.

    See:

        `Evaluation measures (information retrieval). Mean Average Precision`_

        `Mean Average Precision (MAP) For Recommender Systems`_

    .. _`Evaluation measures (information retrieval). Mean Average Precision`:
        https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    .. _`Mean Average Precision (MAP) For Recommender Systems`:
        https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

    Example:
        >>> gt_tops = torch.tensor([
        ...                         [1, 0, 0],
        ...                         [0, 1, 1],
        ...                         [0, 0, 1]
        ... ], dtype=torch.bool)
        >>> n_gt = torch.tensor([2, 3, 5])
        >>> calc_map(gt_tops, n_gt, top_k=(1, 2, 3))
        [tensor([1., 0., 0.]), tensor([0.5000, 0.2500, 0.0000]), tensor([0.5000, 0.3889, 0.1111])]
    """
    _check_if_nonempty_integers_and_positive(top_k, "top_k")
    top_k = _clip_max_with_warning(top_k, gt_tops.shape[1])
    map = []
    correct_preds = torch.cumsum(gt_tops.float(), dim=1)
    for k in top_k:
        _n_gt = torch.min(n_gt, torch.tensor(k).unsqueeze(0))
        positions = torch.arange(1, k + 1).unsqueeze(0)
        map.append(torch.sum((correct_preds[:, :k] / positions) * gt_tops[:, :k], dim=1) / _n_gt)
    return map


def calc_fnmr_at_fmr(pos_dist: torch.Tensor, neg_dist: torch.Tensor, fmr_vals: Tuple[int, ...] = (1,)) -> torch.Tensor:
    """
    Function to compute False Non Match Rate (FNMR) value when False Match Rate (FMR) value
    is equal to ``fmr_vals``.

    The metric calculates the percentage of positive distances higher than a given :math:`q`-th percentile
    of negative distances.

    Args:
        pos_dist: Distances between relevant samples.
        neg_dist: Distances between non-relevant samples.
        fmr_vals: Values of ``fmr`` (measured in percents) for which we compute the corresponding ``fnmr``.
                  For example, if ``fmr_values`` is (20, 40) we will calculate ``fnmr@fmr=20`` and ``fnmr@fmr=40``
    Returns:
        Tensor of ``fnmr@fmr`` values.

    Given a vector of :math:`N` distances between samples from the same classes, :math:`u`,
    the false non-match rate (:math:`\\textrm{FNMR}`) is computed as the proportion below some threshold, :math:`T`:

    .. math::

        \\textrm{FNMR}(T) = \\frac{1}{N}\\sum\\limits_{i = 1}^{N}H\\left(u_i - T\\right) =
        1 - \\frac{1}{N}\\sum\\limits_{i = 1}^{N}H\\left(T - u_i\\right)

    where :math:`H(x)` is the unit step function, and :math:`H(0)` taken to be :math:`1`.

    Similarly, given a vector of :math:`N` distances between samples from different classes, :math:`v`,
    the false match rate (:math:`\\textrm{FMR}`) is computed as the proportion above :math:`T`:

    .. math::

        \\textrm{FMR}(T) = 1 - \\frac{1}{N}\\sum\\limits_{i = 1}^{N}H\\left(v_i - T\\right) =
        \\frac{1}{N}\\sum\\limits_{i = 1}^{N}H\\left(T - v_i\\right)

    Given some interesting false match rate values :math:`\\textrm{FMR}_k` one can find thresholds :math:`T_k`
    corresponding to :math:`\\textrm{FMR}` measurements

    .. math::

        T_k = Q_v\\left(\\textrm{FMR}_k\\right)

    where :math:`Q` is the quantile function, and evaluate the corresponding values of
    :math:`\\textrm{FNMR}@\\textrm{FMR}\\left(T_k\\right) \\stackrel{\\text{def}}{=} \\textrm{FNMR}\\left(T_k\\right)`.


    See:

        `Biometrics Performance`_

        `BIOMETRIC RECOGNITION: A MODERN ERA FOR SECURITY`_

    .. _`Biometrics Performance`:
        https://en.wikipedia.org/wiki/Biometrics#Performance

    .. _`BIOMETRIC RECOGNITION: A MODERN ERA FOR SECURITY`:
        https://www.researchgate.net/publication/50315614_BIOMETRIC_RECOGNITION_A_MODERN_ERA_FOR_SECURITY


    Example:
        >>> pos_dist = torch.tensor([0, 0, 1, 1, 2, 2, 5, 5, 9, 9])
        >>> neg_dist = torch.tensor([3, 3, 4, 4, 6, 6, 7, 7, 8, 8])
        >>> calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals=(10, 50))
        tensor([40., 20.])

    """
    if len(fmr_vals) == 0:
        raise ValueError(f"fmr_vals are expected have at least one value, but got {fmr_vals}")
    if not all(0 <= f <= 100 for f in fmr_vals):
        raise ValueError(f"fmr_vals are expected to be integers in range [0, 100] but got {fmr_vals}")
    thresholds = torch.from_numpy(np.percentile(neg_dist.cpu().numpy(), fmr_vals)).to(pos_dist)
    fnmr_at_fmr = 100 * (pos_dist[None, :] >= thresholds[:, None]).sum(axis=1) / len(pos_dist)
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
    Returns:
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


def _check_if_nonempty_integers_and_positive(seq: Sequence[int], name: str) -> None:
    """
    Check whether ``args`` is not empty and all its elements are positive integers.

    Args:
        seq: A sequence.
        name: A name of the sequence in case of exception should be raised.

    """
    if not len(seq) > 0 or not all([isinstance(x, int) and (x > 0) for x in seq]):
        raise ValueError(f"{name} is expected to be non-empty and contain positive integers, but got {seq}")


def _clip_max_with_warning(arr: Tuple[int, ...], max_el: int) -> Tuple[int, ...]:
    """
    Clip ``arr`` by upper bound ``max_el`` and raise warning if required.

    Args:
        arr: Integer to check and clip.
        max_el: The upper limit.

    Returns:
        Clipped value of ``arr``.
    """
    if any(a > max_el for a in arr):
        warnings.warn(
            f"The desired value of top_k can't be larger than {max_el}, but got {arr}. "
            f"The value of top_k will be clipped to {max_el}."
        )
    return clip_max(arr, max_el)


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
