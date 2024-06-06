import warnings
from copy import deepcopy
from pprint import pprint
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import FloatTensor, LongTensor

from oml.const import (
    CATEGORIES_COLUMN,
    EMBEDDINGS_KEY,
    LOG_TOPK_IMAGES_PER_ROW,
    LOG_TOPK_ROWS_PER_METRIC,
    OVERALL_CATEGORIES_KEY,
)
from oml.ddp.utils import is_main_process
from oml.functional.metrics import (
    TMetricsDict,
    calc_fnmr_at_fmr_by_distances,
    calc_retrieval_metrics,
    calc_topological_metrics,
    reduce_metrics,
)
from oml.interfaces.datasets import IQueryGalleryLabeledDataset, IVisualizableDataset
from oml.interfaces.metrics import IMetricVisualisable, TIndices
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.metrics.accumulation import Accumulator
from oml.retrieval.retrieval_results import RetrievalResults
from oml.utils.misc import flatten_dict, pad_array_right, remove_unused_kwargs


def calc_retrieval_metrics_rr(
    rr: RetrievalResults,
    query_categories: Optional[Union[LongTensor, np.ndarray]] = None,
    cmc_top_k: Tuple[int, ...] = (5,),
    precision_top_k: Tuple[int, ...] = (5,),
    map_top_k: Tuple[int, ...] = (5,),
    reduce: bool = True,
    verbose: bool = True,
) -> TMetricsDict:
    """
    Function to compute different retrieval metrics.

    Args:
        rr: An instance of `RetrievalResults`.
        query_categories: Categories of queries with the size of ``n_query`` to compute metrics for each category.
        cmc_top_k: Values of ``k`` to calculate ``cmc@k`` (`Cumulative Matching Characteristic`)
        precision_top_k: Values of ``k`` to calculate ``precision@k``
        map_top_k: Values of ``k`` to calculate ``map@k`` (`Mean Average Precision`)
        reduce: If ``False`` return metrics for each query without averaging
        verbose: Set ``True`` to make the function verbose.

    Returns:
        Metrics dictionary.

    """
    return calc_retrieval_metrics(
        retrieved_ids=rr.retrieved_ids,
        gt_ids=rr.gt_ids,
        cmc_top_k=cmc_top_k,
        precision_top_k=precision_top_k,
        map_top_k=map_top_k,
        query_categories=query_categories,
        reduce=reduce,
        verbose=verbose,
    )


def calc_fnmr_at_fmr_rr(
    rr: RetrievalResults,
    fmr_vals: Tuple[float, ...] = (0.1,),
) -> TMetricsDict:
    """
    For more details see `calc_fnmr_at_fmr` docs.

    Args:
        rr: An instance of `RetrievalResults`.:
        fmr_vals: Values of `FMR` to calculate `FNMR` at.

    Returns:
        Metrics dictionary.
    """

    max_size = max(len(d) for d in rr.distances)
    dist = np.stack([pad_array_right(np.array(d), max_size, val=-1) for d in rr.distances])

    mask_gt = np.zeros(dist.shape, dtype=bool)
    mask_not_padding = np.ones(dist.shape, dtype=bool)

    for i, (ri, gt_id) in enumerate(zip(rr.retrieved_ids, rr.gt_ids)):
        is_correct = torch.isin(ri, gt_id)
        mask_gt[i, : len(is_correct)] = is_correct
        mask_not_padding[i, len(ri) :] = False

    pos_dist = dist[mask_gt & mask_not_padding].flatten()
    neg_dist = dist[~mask_gt & mask_not_padding].flatten()

    return calc_fnmr_at_fmr_by_distances(pos_dist=pos_dist, neg_dist=neg_dist, fmr_vals=fmr_vals)


class EmbeddingMetrics(IMetricVisualisable):
    """
    This class is designed to accumulate model outputs produced for every batch.
    Since retrieval metrics are not additive, we can compute them only after all data has been collected.

    """

    metric_name = ""

    def __init__(
        self,
        dataset: Optional[IQueryGalleryLabeledDataset],
        cmc_top_k: Tuple[int, ...] = (5,),
        precision_top_k: Tuple[int, ...] = (5,),
        map_top_k: Tuple[int, ...] = (5,),
        fmr_vals: Tuple[float, ...] = tuple(),
        pcf_variance: Tuple[float, ...] = (0.5,),
        postprocessor: Optional[IRetrievalPostprocessor] = None,
        metrics_to_exclude_from_visualization: Iterable[str] = (),
        return_only_overall_category: bool = False,
        visualize_only_overall_category: bool = True,
        verbose: bool = True,
    ):
        """

        Args:
            dataset: Annotated dataset having query-gallery split.
            cmc_top_k: Values of ``k`` to calculate ``cmc@k`` (`Cumulative Matching Characteristic`)
            precision_top_k: Values of ``k`` to calculate ``precision@k``
            map_top_k: Values of ``k`` to calculate ``map@k`` (`Mean Average Precision`)
            fmr_vals: Values of ``fmr`` (measured in quantiles) to calculate ``fnmr@fmr`` (`False Non Match Rate
                      at the given False Match Rate`).
                      For example, if ``fmr_values`` is (0.2, 0.4) we will calculate ``fnmr@fmr=0.2``
                      and ``fnmr@fmr=0.4``.
                      Note, computing this metric requires additional memory overhead,
                      that is why it's turned off by default.
            pcf_variance: Values in range [0, 1]. Find the number of components such that the amount
                          of variance that needs to be explained is greater than the percentage specified
                          by ``pcf_variance``.
            postprocessor: Postprocessor which applies some techniques like query reranking
            metrics_to_exclude_from_visualization: Names of the metrics to exclude from the visualization. It will not
             affect calculations.
            return_only_overall_category: Set ``True`` if you want to return only the aggregated metrics
            visualize_only_overall_category: Set ``False`` if you want to visualize each category separately
            verbose: Set ``True`` if you want to print metrics

        """
        self.dataset = dataset
        self.cmc_top_k = cmc_top_k
        self.precision_top_k = precision_top_k
        self.map_top_k = map_top_k
        self.fmr_vals = fmr_vals
        self.pcf_variance = pcf_variance
        self.postprocessor = postprocessor

        self.retrieval_results: Optional[RetrievalResults] = None

        self.metrics: Optional[TMetricsDict] = None
        self.metrics_unreduced: Optional[TMetricsDict] = None

        self.visualize_only_overall_category = visualize_only_overall_category
        self.return_only_overall_category = return_only_overall_category

        self.metrics_to_exclude_from_visualization = ["fnmr@fmr", "pcf", *metrics_to_exclude_from_visualization]
        self.verbose = verbose

        self._acc_embeddings_key = "__embeddings"
        self.acc = Accumulator(keys_to_accumulate=(self._acc_embeddings_key,))

        if fmr_vals:
            warnings.warn("Note, computing FNMR@FMR may significantly decrease computation time and memory consuming!")

    def setup(self, num_samples: Optional[int] = None) -> None:  # type: ignore
        self.retrieval_results = None
        self.metrics = None
        self.metrics_unreduced = None

        num_samples = num_samples if num_samples is not None else len(self.dataset)
        self.acc.refresh(num_samples)

    def update(self, embeddings: FloatTensor, indices: TIndices) -> None:
        """
        Args:
            embeddings: Representations of the dataset items containing in the current batch.
            indices: Global indices of the dataset items within the range of ``(0, dataset_size - 1)``.
                     Indices are needed to make sure that we can align dataset items and collected information.

        """
        indices = indices if isinstance(indices, List) else indices.tolist()
        self.acc.update_data(data_dict={self._acc_embeddings_key: embeddings}, indices=indices)

    def update_data(self, data: Dict[str, Any], indices: TIndices) -> Any:
        self.update(embeddings=data[EMBEDDINGS_KEY], indices=indices)

    def _compute_retrieval_results(self) -> None:
        # note, fmr requires a lot of compute
        fmr_vals = len(self.dataset.get_gallery_ids()) if self.fmr_vals else 1
        max_k = max([*self.cmc_top_k, *self.precision_top_k, *self.map_top_k, fmr_vals])
        if self.postprocessor:
            # todo: refactor how we deal with postprocessors after we have more examples
            top_n = getattr(self.postprocessor, "top_n", len(self.dataset.get_gallery_ids()))
            max_k = max(max_k, top_n)

        self.retrieval_results = RetrievalResults.from_embeddings(  # type: ignore
            embeddings=self.acc.storage[self._acc_embeddings_key],
            dataset=self.dataset,
            n_items=max_k,
            verbose=self.verbose,
        )

        if self.postprocessor:
            args = {"rr": self.retrieval_results, "dataset": self.dataset}
            args = remove_unused_kwargs(args, self.postprocessor.process)
            self.retrieval_results = self.postprocessor.process(**args)

    def compute_metrics(self) -> TMetricsDict:  # type: ignore
        self.acc = self.acc.sync()  # gathering data from devices happens here if DDP

        if not self.acc.is_storage_full():
            raise ValueError(
                f"Metrics have to be calculated on fully collected data. "
                f"The size of the current storage is less than num samples: "
                f"we've collected {self.acc.collected_samples} out of {self.acc.num_samples}."
            )

        self._compute_retrieval_results()

        args_r = {
            "cmc_top_k": self.cmc_top_k,
            "precision_top_k": self.precision_top_k,
            "map_top_k": self.map_top_k,
            "rr": self.retrieval_results,
            "reduce": False,
            "verbose": self.verbose,
        }

        args_t = {"embeddings": self.acc.storage[self._acc_embeddings_key], "pcf_variance": self.pcf_variance}

        if CATEGORIES_COLUMN in self.dataset.extra_data:
            categories = np.array(self.dataset.extra_data[CATEGORIES_COLUMN])
            query_categories = categories[self.dataset.get_query_ids()]

            metrics_r = calc_retrieval_metrics_rr(query_categories=query_categories, **args_r)  # type: ignore
            metrics_t = calc_topological_metrics(categories=categories, **args_t)  # type: ignore
            self.metrics_unreduced = {cat: {**metrics_r[cat], **metrics_t[cat]} for cat in metrics_r.keys()}

        else:
            metrics_r = calc_retrieval_metrics_rr(**args_r)  # type: ignore
            metrics_t = calc_topological_metrics(**args_t)  # type: ignore
            self.metrics_unreduced = {OVERALL_CATEGORIES_KEY: {**metrics_r, **metrics_t}}

        self.metrics_unreduced[OVERALL_CATEGORIES_KEY].update(
            calc_fnmr_at_fmr_rr(self.retrieval_results, self.fmr_vals)
        )

        self.metrics = reduce_metrics(deepcopy(self.metrics_unreduced))

        if self.return_only_overall_category:
            metric_to_return = {OVERALL_CATEGORIES_KEY: deepcopy(self.metrics[OVERALL_CATEGORIES_KEY])}
        else:
            metric_to_return = deepcopy(self.metrics)

        if self.verbose and is_main_process():
            print("\nMetrics:")
            pprint(metric_to_return)

        return metric_to_return

    def ready_to_visualize(self) -> bool:
        return isinstance(self.dataset, IVisualizableDataset)

    def visualize(self) -> Tuple[Collection[plt.Figure], Collection[str]]:
        """
        Visualize worst queries by all the available metrics.
        """
        metrics_flat = flatten_dict(self.metrics, ignored_keys=self.metrics_to_exclude_from_visualization)
        figures = []
        titles = []
        for metric_name in metrics_flat:
            if self.visualize_only_overall_category and not metric_name.startswith(OVERALL_CATEGORIES_KEY):
                continue
            fig = self.get_plot_for_worst_queries(
                metric_name=metric_name, n_queries=LOG_TOPK_ROWS_PER_METRIC, n_instances=LOG_TOPK_IMAGES_PER_ROW
            )
            log_str = f"top {LOG_TOPK_ROWS_PER_METRIC} worst by {metric_name}".replace("/", "_")
            figures.append(fig)
            titles.append(log_str)
        return figures, titles

    def get_worst_queries_ids(self, metric_name: str, n_queries: int) -> List[int]:
        metric_values = flatten_dict(self.metrics_unreduced)[metric_name]  # type: ignore
        return torch.topk(metric_values, min(n_queries, len(metric_values)), largest=False)[1].tolist()

    def get_plot_for_worst_queries(
        self, metric_name: str, n_queries: int, n_instances: int, verbose: bool = False
    ) -> plt.Figure:
        query_ids = self.get_worst_queries_ids(metric_name=metric_name, n_queries=n_queries)
        return self.get_plot_for_queries(query_ids=query_ids, n_instances=n_instances, verbose=verbose)

    def get_plot_for_queries(self, query_ids: List[int], n_instances: int, verbose: bool = True) -> plt.Figure:
        """
        Args:
            query_ids: Indices of the queries
            n_instances: Amount of the retrieved items to show
            verbose: Set ``True`` for additional information

        """
        assert self.retrieval_results is not None, "We are not ready to plot, because there are no retrieval results."
        assert self.metrics_unreduced is not None, "We are not ready to plot, because metrics were not calculated yet."

        fig = self.retrieval_results.visualize(
            query_ids=query_ids, n_galleries_to_show=n_instances, verbose=verbose, dataset=self.dataset
        )
        fig.tight_layout()
        return fig


__all__ = ["EmbeddingMetrics", "calc_retrieval_metrics_rr", "calc_fnmr_at_fmr_rr"]
