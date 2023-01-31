import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from oml.losses.triplet import get_tri_ids_in_plain
from oml.utils.misc import check_if_nonempty_positive_integers, clip_max
from oml.utils.misc_torch import PCA, elementwise_dist, pairwise_dist, take_2d

TMetricsDict = Dict[str, Dict[Union[int, float], Union[float, Tensor]]]


def calc_retrieval_metrics(
    distances: Tensor,
    mask_gt: Tensor,
    mask_to_ignore: Optional[Tensor] = None,
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
        mask_to_ignore: Binary matrix to indicate that some elements in the gallery cannot be used
                     as answers and must be ignored
        cmc_top_k: Values of ``k`` to calculate ``cmc@k`` (`Cumulative Matching Characteristic`)
        precision_top_k: Values of ``k`` to calculate ``precision@k``
        map_top_k: Values of ``k`` to calculate ``map@k`` (`Mean Average Precision`)
        fmr_vals: Values of ``fmr`` (measured in quantiles) to calculate ``fnmr@fmr`` (`False Non Match Rate
                  at the given False Match Rate`).
                  For example, if ``fmr_values`` is (0.2, 0.4) we will calculate ``fnmr@fmr=0.2`` and ``fnmr@fmr=0.4``
        reduce: If ``False`` return metrics for each query without averaging
        check_dataset_validity: Set ``True`` if you want to check that we have available answers in the gallery for
         each of the queries

    Returns:
        Metrics dictionary.

    """
    top_k_args = [cmc_top_k, precision_top_k, map_top_k]

    if not any(top_k_args + [fmr_vals]):
        raise ValueError("You must specify arguments for at leas 1 metric to calculate it")

    if check_dataset_validity:
        validate_dataset(mask_gt=mask_gt, mask_to_ignore=mask_to_ignore)

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
    gt_tops = take_2d(mask_gt, ii_top_k)
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


def calc_topological_metrics(embeddings: Tensor, pfc_variance: Tuple[float, ...]) -> TMetricsDict:
    """
    Function to evaluate different topological metrics.

    Args:
        embeddings: Embeddings matrix with the shape of ``[n_embeddings, embeddings_dim]``.
        pfc_variance: Values in range [0, 1]. Find the number of components such that the amount
                      of variance that needs to be explained is greater than the percentage specified
                      by ``pfc_variance``.

    Returns:
        Metrics dictionary.

    """
    metrics: TMetricsDict = dict()

    if pfc_variance:
        main_components = calc_pcf(embeddings, pfc_variance)
        metrics["pcf"] = dict(zip(pfc_variance, main_components))

    return metrics


def reduce_metrics(metrics_to_reduce: TMetricsDict) -> TMetricsDict:
    output: TMetricsDict = {}

    for k, v in metrics_to_reduce.items():
        if isinstance(v, (Tensor, np.ndarray)):
            output[k] = v.mean()
        elif isinstance(v, (float, int)):
            output[k] = v
        else:
            output[k] = reduce_metrics(v)  # type: ignore

    return output


def apply_mask_to_ignore(distances: Tensor, mask_gt: Tensor, mask_to_ignore: Tensor) -> Tuple[Tensor, Tensor]:
    distances[mask_to_ignore] = float("inf")
    mask_gt[mask_to_ignore] = False
    return distances, mask_gt


def calc_gt_mask(
    labels: Union[np.ndarray, Tensor], is_query: Union[np.ndarray, Tensor], is_gallery: Union[np.ndarray, Tensor]
) -> Tensor:
    assert all(isinstance(vector, (np.ndarray, Tensor)) for vector in [labels, is_query, is_gallery])
    assert labels.ndim == is_query.ndim == is_gallery.ndim == 1
    assert len(labels) == len(is_query) == len(is_gallery)

    labels, is_query, is_gallery = map(_to_tensor, [labels, is_query, is_gallery])

    query_mask = is_query == 1
    gallery_mask = is_gallery == 1
    query_labels = labels[query_mask]
    gallery_labels = labels[gallery_mask]
    gt_mask = query_labels[..., None] == gallery_labels[None, ...]

    return gt_mask


def calc_mask_to_ignore(is_query: Union[np.ndarray, Tensor], is_gallery: Union[np.ndarray, Tensor]) -> Tensor:
    assert all(isinstance(vector, (np.ndarray, Tensor)) for vector in [is_query, is_gallery])
    assert is_query.ndim == is_gallery.ndim == 1
    assert len(is_query) == len(is_gallery)

    is_query, is_gallery = map(_to_tensor, [is_query, is_gallery])

    ids_query = torch.nonzero(is_query).squeeze()
    ids_gallery = torch.nonzero(is_gallery).squeeze()
    mask_to_ignore = ids_query[..., None] == ids_gallery[None, ...]

    return mask_to_ignore


def calc_distance_matrix(
    embeddings: Union[np.ndarray, Tensor], is_query: Union[np.ndarray, Tensor], is_gallery: Union[np.ndarray, Tensor]
) -> Tensor:
    assert all(isinstance(vector, (np.ndarray, Tensor)) for vector in [embeddings, is_query, is_gallery])
    assert is_query.ndim == 1 and is_gallery.ndim == 1 and embeddings.ndim == 2
    assert embeddings.shape[0] == len(is_query) == len(is_gallery)

    embeddings, is_query, is_gallery = map(_to_tensor, [embeddings, is_query, is_gallery])

    query_mask = is_query == 1
    gallery_mask = is_gallery == 1
    query_embeddings = embeddings[query_mask]
    gallery_embeddings = embeddings[gallery_mask]

    distance_matrix = pairwise_dist(x1=query_embeddings, x2=gallery_embeddings, p=2)

    return distance_matrix


def calculate_accuracy_on_triplets(embeddings: Tensor, reduce_mean: bool = True) -> Tensor:
    assert embeddings.ndim == 2
    assert embeddings.shape[0] % 3 == 0

    anchor_ii, positive_ii, negative_ii = get_tri_ids_in_plain(n=len(embeddings))

    pos_dists = elementwise_dist(x1=embeddings[anchor_ii], x2=embeddings[positive_ii])
    neg_dists = elementwise_dist(x1=embeddings[anchor_ii], x2=embeddings[negative_ii])

    acc = (pos_dists < neg_dists).float()

    if reduce_mean:
        return acc.mean()
    else:
        return acc


def calc_cmc(gt_tops: Tensor, top_k: Tuple[int, ...]) -> List[Tensor]:
    """
    Function to compute Cumulative Matching Characteristics (CMC) at cutoffs ``top_k``.

    ``cmc@k`` for a given query equals to 1 if there is at least 1 instance related to this query in top ``k``
    gallery instances sorted by distances to the query, and 0 otherwise.
    The final ``cmc@k`` could be obtained by averaging the results calculated for each query.

    Args:
        gt_tops: Matrix where the ``(i, j)`` element indicates if ``j``-th gallery sample is related to
                 ``i``-th query or not. Obtained from the full ground truth matrix by taking ``max(top_k)`` elements
                 with the smallest distances to the corresponding queries.
        top_k: Values of ``k`` to calculate ``cmc@k``.

    Returns:
        List of ``cmc@k`` tensors.

    .. math::
        \\textrm{cmc}@k = \\begin{cases}
        1, & \\textrm{if top-}k \\textrm{ ranked gallery samples include an output relevant to the query}, \\\\
        0, & \\textrm{otherwise}.
        \\end{cases}

    Example:
        >>> gt_tops = torch.tensor([
        ...                         [1, 0],
        ...                         [0, 1],
        ...                         [0, 0]
        ... ], dtype=torch.bool)
        >>> calc_cmc(gt_tops, top_k=(1, 2))
        [tensor([1., 0., 0.]), tensor([1., 1., 0.])]
    """
    check_if_nonempty_positive_integers(top_k, "top_k")
    top_k = _clip_max_with_warning(top_k, gt_tops.shape[1])
    cmc = []
    for k in top_k:
        cmc.append(torch.any(gt_tops[:, :k], dim=1).float())
    return cmc


def calc_precision(gt_tops: Tensor, n_gt: Tensor, top_k: Tuple[int, ...]) -> List[Tensor]:
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
        top_k: Values of ``k`` to calculate ``precision@k``.

    Returns:
        List of ``precision@k`` tensors.

    Given a list :math:`g=[g_1, \\ldots, g_k]` of ground truth top :math:`k` closest elements from the gallery to
    a given query (:math:`g_i` is 1 if :math:`i`-th element from the gallery is relevant to the query and 0 otherwise),
    and the total number of relevant elements from the gallery :math:`n`, the :math:`\\textrm{precision}@k`
    for the query is defined as

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
        ...                         [1, 0],
        ...                         [0, 1],
        ...                         [0, 0]
        ... ], dtype=torch.bool)
        >>> n_gt = torch.tensor([2, 3, 5])
        >>> calc_precision(gt_tops, n_gt, top_k=(1, 2))
        [tensor([1., 0., 0.]), tensor([0.5000, 0.5000, 0.0000])]
    """
    check_if_nonempty_positive_integers(top_k, "top_k")
    top_k = _clip_max_with_warning(top_k, gt_tops.shape[1])
    precision = []
    correct_preds = torch.cumsum(gt_tops.float(), dim=1)
    for k in top_k:
        _n_gt = torch.min(n_gt, torch.tensor(k).unsqueeze(0))
        precision.append(correct_preds[:, k - 1] / _n_gt)
    return precision


def calc_map(gt_tops: Tensor, n_gt: Tensor, top_k: Tuple[int, ...]) -> List[Tensor]:
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
        top_k: Values of ``k`` to calculate ``map@k``.

    Returns:
        List of ``map@k`` tensors.

    Given a list :math:`g=[g_1, \\ldots, g_k]` of ground truth top :math:`k` closest elements from the gallery to
    a given query (:math:`g_i` is 1 if :math:`i`-th element from the gallery is relevant to the query and 0 otherwise),
    and the total number of relevant elements from the gallery :math:`n`, the :math:`\\textrm{map}@k`
    for the query is defined as

    .. math::
        \\begin{split}\\textrm{map}@k &=
        \\frac{1}{\\min{\\left(k, n\\right)}}\\sum\\limits_{i = 1}^k
        \\frac{\\textrm{# of relevant elements among top } i\\textrm{ elements}}{i} \\times \\textrm{rel}(i) = \\\\
        & = \\frac{1}{\\min{\\left(k, n\\right)}}\\sum\\limits_{i = 1}^k
        \\frac{\\sum\\limits_{j = 1}^{i}g_j}{i} \\times \\textrm{rel}(i)
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
        ...                         [1, 0],
        ...                         [0, 1],
        ...                         [0, 0]
        ... ], dtype=torch.bool)
        >>> n_gt = torch.tensor([2, 3, 5])
        >>> calc_map(gt_tops, n_gt, top_k=(1, 2))
        [tensor([1., 0., 0.]), tensor([0.5000, 0.2500, 0.0000])]
    """
    check_if_nonempty_positive_integers(top_k, "top_k")
    top_k = _clip_max_with_warning(top_k, gt_tops.shape[1])
    map = []
    correct_preds = torch.cumsum(gt_tops.float(), dim=1)
    for k in top_k:
        _n_gt = torch.min(n_gt, torch.tensor(k).unsqueeze(0))
        positions = torch.arange(1, k + 1).unsqueeze(0)
        map.append(torch.sum((correct_preds[:, :k] / positions) * gt_tops[:, :k], dim=1) / _n_gt)
    return map


def calc_fnmr_at_fmr(pos_dist: Tensor, neg_dist: Tensor, fmr_vals: Tuple[float, ...] = (0.1,)) -> Tensor:
    """
    Function to compute False Non Match Rate (FNMR) value when False Match Rate (FMR) value
    is equal to ``fmr_vals``.

    The metric calculates the quantile of positive distances higher than a given :math:`q`-th quantile
    of negative distances.

    Args:
        pos_dist: Distances between relevant samples.
        neg_dist: Distances between non-relevant samples.
        fmr_vals: Values of ``fmr`` (measured in quantiles) to compute the corresponding ``fnmr``.
                  For example, if ``fmr_values`` is (0.2, 0.4) we will calculate ``fnmr@fmr=0.2`` and ``fnmr@fmr=0.4``

    Returns:
        Tensor of ``fnmr@fmr`` values.

    Given a vector of :math:`N` distances between relevant samples, :math:`u`,
    the false non-match rate (:math:`\\textrm{FNMR}`) is computed as the proportion of :math:`u` below some threshold,
    :math:`T`:

    .. math::

        \\textrm{FNMR}(T) = \\frac{1}{N}\\sum\\limits_{i = 1}^{N}H\\left(u_i - T\\right) =
        1 - \\frac{1}{N}\\sum\\limits_{i = 1}^{N}H\\left(T - u_i\\right)

    where :math:`H(x)` is the unit step function, and :math:`H(0)` taken to be :math:`1`.

    Similarly, given a vector of :math:`N` distances between non-relevant samples, :math:`v`,
    the false match rate (:math:`\\textrm{FMR}`) is computed as the proportion of :math:`v` above some threshold,
    :math:`T`:

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
        >>> calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals=(0.1, 0.5))
        tensor([0.4000, 0.2000])

    """
    _check_if_in_range(fmr_vals, 0, 1, "fmr_vals")
    thresholds = torch.from_numpy(np.quantile(neg_dist.cpu().numpy(), fmr_vals)).to(pos_dist)
    fnmr_at_fmr = (pos_dist[None, :] >= thresholds[:, None]).sum(axis=1) / len(pos_dist)
    return fnmr_at_fmr


def calc_pcf(embeddings: Tensor, pfc_variance: Tuple[float, ...]) -> List[Tensor]:
    """
    Function estimates the Principal Components Fraction (PCF) of embeddings using Principal Component Analysis.
    The metric is defined as a fraction of components needed to explain the required variance in data.

    Args:
        embeddings: Embeddings matrix with the shape of ``[n_embeddings, embeddings_dim]``.
        pfc_variance: Values in range [0, 1]. Find the number of components such that the amount
                      of variance that needs to be explained is greater than the fraction specified
                      by ``pfc_variance``.
    Returns:
        List of linear dimensions as a fractions of the embeddings dimension.

    Let :math:`X` be a set of :math:`d` dimensional embeddings.
    Let :math:`\\lambda_1, \\ldots, \\lambda_d\\in\\mathbb{R}` be a set of eigenvalues
    of the covariance matrix of :math:`X` sorted in descending order.
    Then for a given value of desired explained variance :math:`r`,
    the number of principal components that explaines :math:`r\\cdot 100\\%%` variance is the largest integer
    :math:`n` such that

    .. math::
        \\frac{\\sum\\limits_{i = 1}^{n - 1}\\lambda_i}{\\sum\\limits_{i = 1}^{d}\\lambda_i} \\leq r

    The function returns

    .. math::
        \\frac{n}{d}

    See:

        `Principal Components Analysis`_

    .. _`Principal Components Analysis`:
        https://en.wikipedia.org/wiki/Principal_component_analysis

    Example:
        In the example bellow there are 4 vectors of length 10, and only first 4 dimensions have non-zero values.
        Its covariance matrix will have only 4 eigenvalues that are greater than 0, i.e. there are only 4 principal
        axes. So, in order to keep at least 50% of the information from the set, we need to keep 2 principal
        axes, and in order to keep all the information we need to keep 5 principal axes (one additional axis appears
        because the number of principal axes is superior to the desired explained variance threshold).

        >>> embeddings = torch.eye(4, 10, dtype=torch.float)
        >>> calc_pcf(embeddings, pfc_variance=(0.5, 1))
        tensor([0.2000, 0.5000])

    """
    # The code below mirrors code from scikit-learn repository:
    # https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/decomposition/_pca.py#L491
    _check_if_in_range(pfc_variance, 0, 1, "pfc_variance")
    pca = PCA(embeddings)
    n_components = pca.calc_principal_axes_number(pfc_variance).to(embeddings)
    return n_components / embeddings.shape[1]


def extract_pos_neg_dists(
    distances: Tensor, mask_gt: Tensor, mask_to_ignore: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    """
    Extract distances between relevant samples, and distances between non-relevant samples.

    Args:
        distances: Distance matrix with the shape of ``[query_size, gallery_size]``
        mask_gt: ``(i,j)`` element indicates if for i-th query j-th gallery is the correct prediction
        mask_to_ignore: Binary matrix to indicate that some elements in gallery cannot be used
                     as answers and must be ignored
    Returns:
        pos_dist: Tensor of distances between relevant samples
        neg_dist: Tensor of distances between non-relevant samples
    """
    if mask_to_ignore is not None:
        mask_to_not_ignore = ~mask_to_ignore
        pos_dist = distances[mask_gt & mask_to_not_ignore]
        neg_dist = distances[~mask_gt & mask_to_not_ignore]
    else:
        pos_dist = distances[mask_gt]
        neg_dist = distances[~mask_gt]
    return pos_dist, neg_dist


def validate_dataset(mask_gt: Tensor, mask_to_ignore: Tensor) -> None:
    assert (
        (mask_gt & ~mask_to_ignore).any(1).all()
    ), "There are queries without available correct answers in the gallery!"


def _to_tensor(array: Union[np.ndarray, Tensor]) -> Tensor:
    if isinstance(array, Tensor):
        return array
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise TypeError("Wrong type")


def _clip_max_with_warning(arr: Tuple[int, ...], max_el: int) -> Tuple[int, ...]:
    """
    Clip ``arr`` by upper bound ``max_el`` and raise warning if required.

    Args:
        arr: Array to check and clip.
        max_el: The upper limit.

    Returns:
        Clipped value of ``arr``.
    """
    if any(a > max_el for a in arr):
        warnings.warn(
            f"The desired value of top_k can't be larger than {max_el}, but got {arr}. "
            f"The values of top_k will be clipped to {max_el}."
        )
    return clip_max(arr, max_el)


def _check_if_in_range(vals: Sequence[float], min_: float, max_: float, name: str) -> None:
    """
    Check whether the ``vals`` are in the range ``[min_, max_]``. Throw the ValueError if not.

    Args:
        vals: Sequence to check.
        min_: Minimal value of the range.
        max_: Maximal value of the range.
        name: Name of the variable to throw the ValueError for.
    """
    if len(vals) == 0:
        raise ValueError(f"{name} is expected to be not empty, but got {vals}")
    if not all(min_ <= x <= max_ for x in vals):
        raise ValueError(f"{name} is expected to contain numbers in range [{min_}, {max_}], but got {vals}")


__all__ = [
    "TMetricsDict",
    "calc_retrieval_metrics",
    "calc_topological_metrics",
    "apply_mask_to_ignore",
    "calc_gt_mask",
    "calc_mask_to_ignore",
    "calc_distance_matrix",
    "calculate_accuracy_on_triplets",
    "reduce_metrics",
]
