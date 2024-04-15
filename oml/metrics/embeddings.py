from copy import deepcopy
from pprint import pprint
from typing import Collection, Dict, Iterable, List, Optional, Tuple, Union

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
    TMetricsDict,
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
from oml.utils.misc import flatten_dict, take_from_list

TMetricsDict_ByLabels = Dict[Union[str, int], TMetricsDict]

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
        dataset: Optional[IDatasetQueryGallery] = None,
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

        self.prediction = None
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

    def setup(self, num_samples: int) -> None:
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
            distances, retrieved_ids = self.postprocessor.process(
                distances=self.prediction.distances, retrieved_ids=self.prediction.retrieved_ids, dataset=self.dataset
            )
            self.prediction = RetrievalPrediction(
                distances=distances, retrieved_ids=retrieved_ids, gt_ids=self.prediction.gt_ids
            )

    def compute_metrics(self) -> TMetricsDict_ByLabels:
        self._obtain_prediction()

        metrics: TMetricsDict_ByLabels = dict()

        metrics[self.overall_categories_key] = calc_retrieval_metrics(
            retrieved_ids=self.prediction.retrieved_ids,
            gt_ids=self.prediction.gt_ids,
            reduce=False,
            cmc_top_k=self.cmc_top_k,
            precision_top_k=self.precision_top_k,
            map_top_k=self.map_top_k,
        )

        embeddings = self.acc.storage[self._embeddings_acc_key]

        metrics[self.overall_categories_key].update(
            calc_topological_metrics(embeddings.float(), pcf_variance=self.pcf_variance)  # type: ignore
        )

        # todo 522: it's bad to calc this metrics not on the full matrix of distances
        metrics[self.overall_categories_key].update(
            calc_fnmr_at_fmr_from_list_of_gt(self.prediction.distances, self.prediction.gt_ids, self.fmr_vals)
        )

        # todo 522: handle categories it properly
        if self.dataset.categories_key:
            categories = np.array(self.dataset.get_categories())
            ids_query = self.dataset.get_query_ids()
            query_categories = categories[ids_query]

            for category in np.unique(query_categories):
                ids_category = (query_categories == category).nonzero()[0]
                print(ids_category)

                metrics[category] = take_unreduced_metrics_by_ids(metrics[self.overall_categories_key], ids_category)

                metrics[category].update(
                    calc_fnmr_at_fmr_from_list_of_gt(
                        self.prediction.distances[ids_category],
                        take_from_list(self.prediction.gt_ids, ids_category),
                        self.fmr_vals,
                    )
                )

                mask = categories == category
                metrics[category].update(calc_topological_metrics(embeddings[mask], pcf_variance=self.pcf_variance))

        # todo 522: confusing names
        self.metrics_unreduced = metrics  # type: ignore
        # note, here we do micro averaging
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

    def ready_to_visualize(self) -> bool:
        return self.prediction is not None

    def get_worst_queries_ids(self, metric_name: str, n_queries: int) -> List[int]:
        if any(nm in metric_name for nm in GLOBAL_METRICS):
            raise ValueError(f"Provided metric {metric_name} cannot be calculated on query level!")

        metric_values = flatten_dict(self.metrics_unreduced)[metric_name]  # type: ignore
        return torch.topk(metric_values, min(n_queries, len(metric_values)), largest=False)[1].tolist()

    def get_plot_for_worst_queries(
        self, metric_name: str, n_queries: int, n_instances: int, verbose: bool = False
    ) -> plt.Figure:
        query_ids = self.get_worst_queries_ids(metric_name=metric_name, n_queries=n_queries)
        return self.get_plot_for_queries(query_ids=query_ids, n_instances=n_instances, verbose=verbose)

    def get_plot_for_queries(self, query_ids: List[int], n_instances: int, verbose: bool = True) -> plt.Figure:
        """
        Visualize the predictions for the query with the indicies <query_ids>.

        Args:
            query_ids: Index of the query
            n_instances: Amount of the predictions to show
            verbose: wether to show image paths or not

        """
        pass
        # todo 522: it has to be implemented in RetrievalPrediction


class EmbeddingMetricsDDP(EmbeddingMetrics, IMetricDDP):
    def sync(self) -> None:
        self.acc = self.acc.sync()


__all__ = ["TMetricsDict_ByLabels", "EmbeddingMetrics", "EmbeddingMetricsDDP"]
