from copy import deepcopy
from pprint import pprint
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import FloatTensor

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
    calc_retrieval_metrics,
    calc_topological_metrics,
    reduce_metrics,
    take_unreduced_metrics_by_mask,
)
from oml.interfaces.datasets import IQueryGalleryLabeledDataset, IVisualizableDataset
from oml.interfaces.metrics import IMetricVisualisable, TIndices
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.metrics.accumulation import Accumulator
from oml.retrieval.retrieval_results import RetrievalResults
from oml.utils.misc import flatten_dict

TMetricsDict_ByLabels = Dict[Union[str, int], TMetricsDict]


def calc_retrieval_metrics_rr(
    rr: RetrievalResults,
    cmc_top_k: Tuple[int, ...] = (5,),
    precision_top_k: Tuple[int, ...] = (5,),
    map_top_k: Tuple[int, ...] = (5,),
    reduce: bool = True,
) -> TMetricsDict:
    return calc_retrieval_metrics(
        retrieved_ids=rr.retrieved_ids,
        gt_ids=rr.gt_ids,
        cmc_top_k=cmc_top_k,
        precision_top_k=precision_top_k,
        map_top_k=map_top_k,
        reduce=reduce,
    )


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

        self.prediction = None

        self.metrics = None
        self.metrics_unreduced = None

        self.visualize_only_overall_category = visualize_only_overall_category
        self.return_only_overall_category = return_only_overall_category

        self.metrics_to_exclude_from_visualization = ["fnmr@fmr", "pcf", *metrics_to_exclude_from_visualization]
        self.verbose = verbose

        self._acc_embeddings_key = "__embeddings"
        self.acc = Accumulator(keys_to_accumulate=(self._acc_embeddings_key,))

    def setup(self, num_samples: Optional[int] = None) -> None:  # type: ignore
        self.prediction = None
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
        max_k = max([*self.cmc_top_k, *self.precision_top_k, *self.map_top_k])

        self.retrieval_results = RetrievalResults.compute_from_embeddings(
            embeddings=self.acc.storage[self._acc_embeddings_key],
            dataset=self.dataset,
            n_items_to_retrieve=max_k,
        )

        if self.postprocessor:
            self.retrieval_results = self.postprocessor.process(self.retrieval_results, self.dataset)

    def compute_metrics(self) -> TMetricsDict_ByLabels:  # type: ignore
        self.acc = self.acc.sync()

        # todo 522: what do we do with fnmr?

        if not self.acc.is_storage_full():
            raise ValueError(
                f"Metrics have to be calculated on fully collected data. "
                f"The size of the current storage is less than num samples: "
                f"we've collected {self.acc.collected_samples} out of {self.acc.num_samples}."
            )

        self._compute_retrieval_results()

        metrics: TMetricsDict_ByLabels = dict()

        # note, here we do micro averaging
        metrics[self.overall_categories_key] = calc_retrieval_metrics_rr(
            rr=self.retrieval_results,
            cmc_top_k=self.cmc_top_k,
            precision_top_k=self.precision_top_k,
            map_top_k=self.map_top_k,
            reduce=False,
        )

        embeddings = self.acc.storage[self._acc_embeddings_key]
        metrics[self.overall_categories_key].update(calc_topological_metrics(embeddings, self.pcf_variance))

        if CATEGORIES_COLUMN in self.dataset.extra_data:
            categories = np.array(self.dataset.extra_data[CATEGORIES_COLUMN])
            ids_query = self.dataset.get_query_ids()
            query_categories = categories[ids_query]

            for category in np.unique(query_categories):
                mask_query_sz = query_categories == category
                metrics[category] = take_unreduced_metrics_by_mask(metrics[self.overall_categories_key], mask_query_sz)

                mask_dataset_sz = categories == category
                metrics[category].update(calc_topological_metrics(embeddings[mask_dataset_sz], self.pcf_variance))

        self.metrics_unreduced = metrics  # type: ignore
        self.metrics = reduce_metrics(metrics)  # type: ignore

        if self.return_only_overall_category:
            metric_to_return = {
                self.overall_categories_key: deepcopy(self.metrics[self.overall_categories_key])  # type: ignore
            }
        else:
            metric_to_return = deepcopy(self.metrics)

        if self.verbose and is_main_process():
            print("\nMetrics:")
            pprint(metric_to_return)

        return metric_to_return  # type: ignore

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
            n_instances: Amount of the predictions to show
            verbose: Set ``True`` for additional information

        """
        assert self.retrieval_results is not None, "We are not ready to plot, because there are no retrieval results."
        assert self.metrics_unreduced is not None, "We are not ready to plot, because metrics were not calculated yet."

        fig = self.retrieval_results.visualize(
            query_ids=query_ids, n_galleries_to_show=n_instances, verbose=verbose, dataset=self.dataset
        )
        fig.tight_layout()
        return fig


__all__ = ["TMetricsDict_ByLabels", "EmbeddingMetrics", "calc_retrieval_metrics_rr"]
