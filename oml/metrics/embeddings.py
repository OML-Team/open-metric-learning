from copy import deepcopy
from pprint import pprint
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, overload

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import FloatTensor

from oml.const import (
    CATEGORIES_COLUMN,
    LOG_TOPK_IMAGES_PER_ROW,
    LOG_TOPK_ROWS_PER_METRIC,
    OVERALL_CATEGORIES_KEY,
)
from oml.ddp.utils import is_main_process
from oml.functional.metrics import (
    calc_fnmr_at_fmr_from_list_of_gt,
    calc_retrieval_metrics,
    calc_topological_metrics,
    reduce_metrics,
    take_unreduced_metrics_by_ids,
)
from oml.interfaces.datasets import IDatasetQueryGallery
from oml.interfaces.metrics import IBasicMetric, IMetricDDP, IMetricVisualizable
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.metrics.accumulation import Accumulator
from oml.retrieval.prediction import RetrievalPrediction
from oml.utils.misc import flatten_dict

GLOBAL_METRICS = ["fnmr@fmr", "pcf"]


class EmbeddingMetrics(IBasicMetric, IMetricVisualizable):
    """
    This class accumulates the information from the batches and embeddings produced by the model
    at every batch in epoch. After all the samples have been stored, you can call the function
    which computes retrievals metrics. To get the needed information from the batches, it uses
    keys which have to be provided as init arguments. Please, check the usage example in
    `Readme`.

    """

    metric_name = ""

    def __init__(
        self,
        dataset: IDatasetQueryGallery,
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
            dataset: Dataset is needed for visualisation and dataset-based postprocessing.
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
            metrics_to_exclude_from_visualization: Names of the metrics to exclude from the visualisation. It will not
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

        self.prediction: Optional[RetrievalPrediction] = None
        self.metrics = None
        self.metrics_unreduced = None

        self.visualize_only_overall_category = visualize_only_overall_category
        self.return_only_overall_category = return_only_overall_category

        self.metrics_to_exclude_from_visualization = [
            GLOBAL_METRICS,  # because this metrics are global, not query-wise
            *metrics_to_exclude_from_visualization,
        ]
        self.verbose = verbose

        self._embeddings_acc_key = "embeddings"
        self.acc = Accumulator(keys_to_accumulate=[self._embeddings_acc_key])

    def setup(self, num_samples: int) -> None:  # type: ignore
        self.prediction = None
        self.metrics = None

        self.acc.refresh(num_samples=num_samples)

    def update_data(self, embeddings: FloatTensor) -> None:
        self.acc.update_data(data_dict={self._embeddings_acc_key: embeddings})

    def _obtain_prediction(self) -> None:
        if not self.acc.is_storage_full():
            raise ValueError(
                f"Metrics have to be calculated on fully collected data. "
                f"The size of the current storage is less than num samples: "
                f"we've collected {self.acc.collected_samples} out of {self.acc.num_samples}."
            )

        max_metrics_k = max([*self.cmc_top_k, *self.precision_top_k, *self.map_top_k])
        gallery_size = len(self.dataset.get_gallery_ids())

        self.prediction = RetrievalPrediction.compute_from_embeddings(
            embeddings=self.acc.storage[self._embeddings_acc_key].float(),
            dataset=self.dataset,
            n_ids_to_retrieve=min(max_metrics_k + 100, gallery_size),
        )

        if self.postprocessor:
            self.prediction = self.postprocessor.process(self.prediction, dataset=self.dataset)

    def compute_metrics(self) -> Dict[str, Any]:
        self._obtain_prediction()

        metrics_unr = dict()

        metrics_unr[self.overall_categories_key] = calc_retrieval_metrics(
            retrieved_ids=self.prediction.retrieved_ids,
            gt_ids=self.prediction.gt_ids,
            reduce=False,
            cmc_top_k=self.cmc_top_k,
            precision_top_k=self.precision_top_k,
            map_top_k=self.map_top_k,
        )

        embeddings = self.acc.storage[self._embeddings_acc_key]

        metrics_unr[self.overall_categories_key].update(
            calc_topological_metrics(embeddings.float(), pcf_variance=self.pcf_variance)
        )

        metrics_unr[self.overall_categories_key].update(
            calc_fnmr_at_fmr_from_list_of_gt(
                self.prediction.distances, self.prediction.gt_ids, self.fmr_vals, len(self.dataset.get_query_ids())
            )
        )

        categories = self.dataset.extra_data.get(CATEGORIES_COLUMN, None)
        if categories is not None:
            ids_query = self.dataset.get_query_ids()
            queries_categories = categories[ids_query]

            for category in np.unique(queries_categories):
                ids_category = (queries_categories == category).nonzero()[0]

                metrics_unr[category] = take_unreduced_metrics_by_ids(
                    metrics_unr[self.overall_categories_key], ids_category
                )

                metrics_unr[category].update(
                    calc_fnmr_at_fmr_from_list_of_gt(
                        self.prediction.distances[ids_category],
                        [self.prediction.gt_ids[i] for i in ids_category],
                        self.fmr_vals,
                        len(self.dataset.get_query_ids()),
                    )
                )

                mask = categories == category
                metrics_unr[category].update(calc_topological_metrics(embeddings[mask], pcf_variance=self.pcf_variance))

        self.metrics_unreduced = metrics_unr
        self.metrics = reduce_metrics(metrics_unr)

        if self.return_only_overall_category:
            metric_to_return = {self.overall_categories_key: deepcopy(self.metrics[self.overall_categories_key])}
        else:
            metric_to_return = deepcopy(self.metrics)

        if self.verbose and is_main_process():
            print("\nMetrics:")
            pprint(metric_to_return)

        return metric_to_return

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
        if any(nm in metric_name for nm in GLOBAL_METRICS):
            raise ValueError(f"Provided metric {metric_name} cannot be calculated on query level!")

        metric_values = flatten_dict(self.metrics_unreduced)[metric_name]
        return torch.topk(metric_values, min(n_queries, len(metric_values)), largest=False)[1].tolist()

    def get_plot_for_worst_queries(
        self, metric_name: str, n_queries: int, n_instances: int, verbose: bool = False
    ) -> plt.Figure:
        query_ids = self.get_worst_queries_ids(metric_name=metric_name, n_queries=n_queries)
        return self.get_plot_for_queries(query_ids=query_ids, n_instances=n_instances, verbose=verbose)

    def get_plot_for_queries(self, query_ids: List[int], n_instances: int, verbose: bool = True) -> plt.Figure:
        if not isinstance(self.dataset, IMetricVisualizable):
            raise ValueError(
                f"The visualisation is only available for {IMetricVisualizable.__name__},"
                f"provided dataset has the type of {type(self.dataset)}."
            )

        return self.prediction.visualize(query_ids, n_instances, ataset=self.dataset, verbose=verbose)


class EmbeddingMetricsDDP(EmbeddingMetrics, IMetricDDP):
    def sync(self) -> None:
        self.acc = self.acc.sync()


__all__ = ["EmbeddingMetrics", "EmbeddingMetricsDDP"]
