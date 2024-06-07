from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, isin
from tqdm.auto import tqdm

from oml.const import OVERALL_CATEGORIES_KEY
from oml.utils.misc import check_if_nonempty_positive_integers
from oml.utils.misc_torch import PCA

TMetricsDict = Dict[str, Any]
TCategories = Union[LongTensor, np.ndarray]


def calc_retrieval_metrics(
    retrieved_ids: Sequence[LongTensor],
    gt_ids: Sequence[LongTensor],
    query_categories: Optional[TCategories] = None,
    cmc_top_k: Tuple[int, ...] = (5,),
    precision_top_k: Tuple[int, ...] = (5,),
    map_top_k: Tuple[int, ...] = (5,),
    reduce: bool = True,
    verbose: bool = True,
) -> TMetricsDict:
    """
    Function to compute different retrieval metrics.

    Args:
        retrieved_ids: First gallery indices retrieved for every query with the size of ``n_query``.
            Every index is within the range ``(0, n_gallery - 1)``.
        gt_ids: Gallery indices relevant to every query with the size of ``n_query``.
            Every element is within the range ``(0, n_gallery - 1)``
        query_categories: Categories of queries with the size of ``n_query`` to compute metrics for each category.
        cmc_top_k: Values of ``k`` to calculate ``cmc@k`` (`Cumulative Matching Characteristic`)
        precision_top_k: Values of ``k`` to calculate ``precision@k``
        map_top_k: Values of ``k`` to calculate ``map@k`` (`Mean Average Precision`)
        reduce: If ``False`` return metrics for each query without averaging
        verbose: Set ``True`` to make the function verbose.

    Returns:
        Metrics dictionary.

    """
    assert len(retrieved_ids) == len(gt_ids)
    assert (query_categories is None) or (len(query_categories) == len(retrieved_ids))

    # let's mark every correctly retrieved item as True and vice versa
    gt_tops = tuple([isin(r, g).bool() for r, g in zip(retrieved_ids, gt_ids)])
    n_gts = [len(ids) for ids in gt_ids]

    metrics: TMetricsDict = defaultdict(dict)

    if cmc_top_k:
        cmc = calc_cmc(gt_tops, n_gts, cmc_top_k, verbose=verbose)
        metrics["cmc"] = dict(zip(cmc_top_k, cmc))

    if precision_top_k:
        precision = calc_precision(gt_tops, n_gts, precision_top_k, verbose=verbose)
        metrics["precision"] = dict(zip(precision_top_k, precision))

    if map_top_k:
        map_ = calc_map(gt_tops, n_gts, map_top_k, verbose=verbose)
        metrics["map"] = dict(zip(map_top_k, map_))

    if query_categories is not None:
        metrics_cat = {c: take_unreduced_metrics_by_mask(metrics, query_categories == c) for c in query_categories}
        metrics = {OVERALL_CATEGORIES_KEY: metrics, **metrics_cat}

    if reduce:
        metrics = reduce_metrics(metrics)

    return metrics


def calc_topological_metrics(
    embeddings: Tensor, pcf_variance: Tuple[float, ...], categories: Optional[TCategories] = None, verbose: bool = False
) -> TMetricsDict:
    """
    Function to evaluate different topological metrics.

    Args:
        embeddings: Embeddings matrix with the shape of ``[n_embeddings, embeddings_dim]``.
        categories: Categories of embeddings to compute category wise metrics.
        pcf_variance: Values in range [0, 1]. Find the number of components such that the amount
                      of variance that needs to be explained is greater than the percentage specified
                      by ``pcf_variance``.
        verbose: Set ``True`` to see a progress bar.

    Returns:
        Metrics dictionary.

    """
    assert (categories is None) or (len(categories) == len(embeddings))

    metrics: TMetricsDict = defaultdict(dict)

    if pcf_variance:
        main_components = calc_pcf(embeddings, pcf_variance)
        metrics["pcf"] = dict(zip(pcf_variance, main_components))

    if pcf_variance and (categories is not None):
        categories_unq = np.unique(categories)
        data = tqdm(categories_unq, desc="Topologic metrics on different categories") if verbose else categories_unq
        metrics_cat = {
            c: calc_topological_metrics(embeddings[categories == c], pcf_variance, categories=None) for c in data
        }
        metrics = {OVERALL_CATEGORIES_KEY: metrics, **metrics_cat}

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


def take_unreduced_metrics_by_mask(metrics: TMetricsDict, mask: BoolTensor) -> TMetricsDict:
    output: TMetricsDict = {}

    for k, v in metrics.items():
        if isinstance(v, Tensor):
            output[k] = v[mask] if v.numel() > 1 else v
        elif isinstance(v, (float, int)):
            output[k] = v
        else:
            output[k] = take_unreduced_metrics_by_mask(v, mask)  # type: ignore

    return output


def calc_cmc(
    gt_tops: Sequence[BoolTensor], n_gts: List[int], top_k: Tuple[int, ...], verbose: bool = False
) -> List[FloatTensor]:
    """
    Function to compute Cumulative Matching Characteristics (CMC) at cutoffs ``top_k``.

    ``cmc@k`` for a given query equals to 1 if there is at least 1 instance related to this query in top ``k``
    gallery instances sorted by distances to the query, and 0 otherwise.
    The final ``cmc@k`` could be obtained by averaging the results calculated for each query.

    Args:
        gt_tops: Indicators that show if retrievied items are correct or not:
            ``gt_tops[i][j]`` is ``True`` if ``j``-th gallery item is related to the ``i``-th query item.
        n_gts: Number of existing ground truths for every query.
        top_k: Values of ``k`` to calculate ``cmc@k``.
        verbose: Set ``True`` to see progress bar.

    Returns:
        List of ``cmc@k`` tensors computed for every query.

    .. math::
        \\textrm{cmc}@k = \\begin{cases}
        1, & \\textrm{if top-}k \\textrm{ ranked gallery samples include an output relevant to the query}, \\\\
        0, & \\textrm{otherwise}.
        \\end{cases}

    Example:
        >>> gt_tops = [
        ...     BoolTensor([1, 0]),
        ...     BoolTensor([0, 1, 1]),
        ...     BoolTensor([0, 0]),
        ...     BoolTensor([])
        ... ]
        >>> n_gts = [2, 2, 1, 0]
        >>> calc_cmc(gt_tops, n_gts, top_k=(1, 2))
        [tensor([1., 0., 0., 1.]), tensor([1., 1., 0., 1.])]
    """
    check_if_nonempty_positive_integers(top_k, "top_k")

    def cmc_single(is_correct: BoolTensor, n_gt: int, k_: int) -> float:
        if n_gt == 0 and len(is_correct) == 0:
            return 1.0
        elif n_gt > 0 and len(is_correct) == 0:
            return 0.0
        else:
            value = float(is_correct[:k_].any())
            return value

    cmc = []
    for k in top_k:
        items = tqdm(zip(gt_tops, n_gts), desc=f"CMC@{k}", total=len(gt_tops)) if verbose else zip(gt_tops, n_gts)
        cmc.append(FloatTensor([cmc_single(gts, n_gt, k) for gts, n_gt in items]))

    return cmc


def calc_precision(
    gt_tops: Sequence[BoolTensor], n_gts: List[int], top_k: Tuple[int, ...], verbose: bool = False
) -> List[FloatTensor]:
    """
    Function to compute Precision at cutoffs ``top_k``.

    ``precision@k`` for a given query is a fraction of the relevant gallery instances among the top ``k`` instances
    sorted by distances from the query to the gallery.
    The final ``precision@k`` could be obtained by averaging the results calculated for each query.

    Args:
        gt_tops: Indicators that show if retrievied items are correct or not:
            ``gt_tops[i][j]`` is ``True`` if ``j``-th gallery item is related to the ``i``-th query item.
        n_gts: Number of existing ground truth for every query.
        top_k: Values of ``k`` to calculate ``precision@k``.
        verbose: Set ``True`` to see progress bar.

    Returns:
        List of ``precision@k`` tensors computed for every query.

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

        >>> gt_tops = [
        ...     BoolTensor([1, 0]),
        ...     BoolTensor([0, 1, 1]),
        ...     BoolTensor([0, 0]),
        ...     BoolTensor([])
        ... ]
        >>> n_gts = [2, 3, 5, 2]
        >>> calc_precision(gt_tops, n_gts, top_k=(1, 2))
        [tensor([1., 0., 0., 0.]), tensor([0.5000, 0.5000, 0.0000, 0.0000])]

    """
    check_if_nonempty_positive_integers(top_k, "top_k")

    def precision_single(is_correct: BoolTensor, n_gt: int, k_: int) -> float:
        if n_gt == 0 and len(is_correct) == 0:
            return 1.0
        elif n_gt > 0 and len(is_correct) == 0:
            return 0.0
        else:
            k_ = min(k_, len(is_correct))
            value = torch.cumsum(is_correct, dim=0)[k_ - 1] / min(n_gt, k_)
            return float(value)

    precision = []
    for k in top_k:
        items = tqdm(zip(gt_tops, n_gts), desc=f"Precision@{k}", total=len(n_gts)) if verbose else zip(gt_tops, n_gts)
        precision.append(FloatTensor([precision_single(gts, n_gt, k) for gts, n_gt in items]))

    return precision


def calc_map(
    gt_tops: Sequence[BoolTensor], n_gts: List[int], top_k: Tuple[int, ...], verbose: bool = False
) -> List[FloatTensor]:
    """
    Function to compute Mean Average Precision (MAP) at cutoffs ``top_k``.

    ``map@k`` for a given query is the average value of the ``precision`` considered as a function of the ``recall``.
    The final ``map@k`` could be obtained by averaging the results calculated for each query.

    Args:
        gt_tops: Indicators that show if retrievied items are correct or not:
            ``gt_tops[i][j]`` is ``True`` if ``j``-th gallery item is related to the ``i``-th query item.
        n_gts: Number of existing ground truth for every query.
        top_k: Values of ``k`` to calculate ``map@k``.
        verbose: Set ``True`` to see progress bar.

    Returns:
        List of ``map@k`` tensors computed for every query.

    Given a list :math:`g=[g_1, \\ldots, g_k]` of ground truth top :math:`k` closest elements from the gallery to
    a given query (:math:`g_i` is 1 if :math:`i`-th element from the gallery is relevant to the query and 0 otherwise),
    and the total number of relevant elements from the gallery :math:`n`, the :math:`\\textrm{map}@k`
    for the query is defined as

    .. math::
        \\begin{split}
        \\textrm{map}@k &=
        \\frac{1}{n_k}\\sum\\limits_{i = 1}^k
        \\frac{n_i}{i} \\times \\textrm{rel}(i)
        \\end{split}

    where :math:`\\textrm{rel}(i)` is 1 if :math:`i`-th element from the top :math:`i` closest
    elements from the gallery to the query is relevant to the query, and 0 otherwise;
    and :math:`n_i = \\sum\\limits_{j = 1}^{i}g_j`, which is the number of the relevant predictions
    among the first :math:`i` outputs.

    See:

        `Evaluation measures (information retrieval). Mean Average Precision`_

        `Mean Average Precision (MAP) For Recommender Systems`_

    .. _`Evaluation measures (information retrieval). Mean Average Precision`:
        https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    .. _`Mean Average Precision (MAP) For Recommender Systems`:
        https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html

    Example:
        >>> gt_tops = [
        ...    BoolTensor([1, 0]),
        ...    BoolTensor([0, 1]),
        ...    BoolTensor([0, 0, 0, 0]),
        ...    BoolTensor([])
        ... ]
        >>> n_gts = [1, 1, 2, 0]
        >>> calc_map(gt_tops, n_gts, top_k=(1, 2))
        [tensor([1., 0., 0., 1.]), tensor([1.0000, 0.5000, 0.0000, 1.0000])]
    """
    check_if_nonempty_positive_integers(top_k, "top_k")

    def map_single(is_correct: BoolTensor, n_gt: int, k_: int) -> float:
        if n_gt == 0 and len(is_correct) == 0:
            return 1.0
        elif n_gt > 0 and len(is_correct) == 0:
            return 0.0
        else:
            k_ = min(k_, len(is_correct))
            correct_preds = torch.cumsum(is_correct, dim=0).float()
            positions = torch.arange(1, k_ + 1).to(correct_preds.device)
            n_k = correct_preds[k_ - 1].clone()
            n_k[n_k < 1] = torch.inf  # hack to avoid zero division
            value = torch.sum((correct_preds[:k_] / positions) * is_correct[:k_], dim=0) / n_k
            return float(value)

    map_ = []
    for k in top_k:
        items = tqdm(zip(gt_tops, n_gts), total=len(gt_tops), desc=f"MAP@{k}") if verbose else zip(gt_tops, n_gts)
        map_.append(FloatTensor([map_single(is_correct, n_gt, k) for is_correct, n_gt in items]))

    return map_


def calc_fnmr_at_fmr(pos_dist: np.ndarray, neg_dist: np.ndarray, fmr_vals: Tuple[float, ...] = (0.1,)) -> FloatTensor:
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
        >>> pos_dist = np.array([0, 0, 1, 1, 2, 2, 5, 5, 9, 9])
        >>> neg_dist = np.array([3, 3, 4, 4, 6, 6, 7, 7, 8, 8])
        >>> calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals=(0.1, 0.5))
        tensor([0.4000, 0.2000])

    """
    _check_if_in_range(fmr_vals, 0, 1, "fmr_vals")
    thresholds = np.quantile(neg_dist, fmr_vals)  # we use numpy because it can take bigger input
    fnmr_at_fmr = (pos_dist[None, :] >= thresholds[:, None]).sum(axis=1) / len(pos_dist)
    return FloatTensor(fnmr_at_fmr)


def calc_fnmr_at_fmr_by_distances(
    pos_dist: np.ndarray, neg_dist: np.ndarray, fmr_vals: Tuple[float, ...]
) -> TMetricsDict:
    metrics: TMetricsDict = dict()

    if fmr_vals:
        fnmr_at_fmr = calc_fnmr_at_fmr(pos_dist, neg_dist, fmr_vals)
        metrics["fnmr@fmr"] = dict(zip(fmr_vals, fnmr_at_fmr))

    return metrics


def calc_pcf(embeddings: Tensor, pcf_variance: Tuple[float, ...]) -> List[Tensor]:
    """
    Function estimates the Principal Components Fraction (PCF) of embeddings using Principal Component Analysis.
    The metric is defined as a fraction of components needed to explain the required variance in data.

    Args:
        embeddings: Embeddings matrix with the shape of ``[n_embeddings, embeddings_dim]``.
        pcf_variance: Values in range [0, 1]. Find the number of components such that the amount
                      of variance that needs to be explained is greater than the fraction specified
                      by ``pcf_variance``.
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
        >>> calc_pcf(embeddings, pcf_variance=(0.5, 1))
        tensor([0.2000, 0.5000])

    """
    # The code below mirrors code from scikit-learn repository:
    # https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b6/sklearn/decomposition/_pca.py#L491
    _check_if_in_range(pcf_variance, 0, 1, "pcf_variance")
    try:
        pca = PCA(embeddings)
        n_components = pca.calc_principal_axes_number(pcf_variance).to(embeddings)
        metric = n_components / embeddings.shape[1]
    except Exception:
        # Mostly we handle the following error here:
        # >>> The algorithm failed to converge because the input matrix is ill-conditioned
        # >>> or has too many repeated singular values
        metric = [torch.tensor(float("nan"))] * len(pcf_variance)

    return metric


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
    "reduce_metrics",
    "take_unreduced_metrics_by_mask",
    "calc_fnmr_at_fmr",
]
