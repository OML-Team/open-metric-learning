from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from oml.const import (
    BLUE,
    EMBEDDINGS_KEY,
    GRAY,
    GREEN,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    LOG_TOPK_IMAGES_PER_ROW,
    LOG_TOPK_ROWS_PER_METRIC,
    N_GT_SHOW_EMBEDDING_METRICS,
    OVERALL_CATEGORIES_KEY,
    PATHS_KEY,
    RED,
    X1_KEY,
    X2_KEY,
    Y1_KEY,
    Y2_KEY,
)
from oml.functional.metrics import (
    TMetricsDict,
    calc_distance_matrix,
    calc_gt_mask,
    calc_mask_to_ignore,
    calc_retrieval_metrics,
    reduce_metrics,
)
from oml.interfaces.metrics import (
    IBasicMetricDDP,
    IBasicMetricDDPWithVisualization,
    IBasicMetricWithVisualization,
)
from oml.interfaces.post_processor import IPostprocessor
from oml.metrics.accumulation import Accumulator
from oml.utils.images.images import get_img_with_bbox, square_pad
from oml.utils.misc import flatten_dict

TMetricsDict_ByLabels = Dict[Union[str, int], TMetricsDict]


class EmbeddingMetrics(IBasicMetricWithVisualization):
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
        embeddings_key: str = EMBEDDINGS_KEY,
        labels_key: str = LABELS_KEY,
        is_query_key: str = IS_QUERY_KEY,
        is_gallery_key: str = IS_GALLERY_KEY,
        extra_keys: Tuple[str, ...] = (),
        cmc_top_k: Tuple[int, ...] = (5,),
        precision_top_k: Tuple[int, ...] = (5,),
        map_top_k: Tuple[int, ...] = (5,),
        categories_key: Optional[str] = None,
        postprocessor: Optional[IPostprocessor] = None,
        visualization_metrics_to_ignore: Iterable[str] = (),
        check_dataset_validity: bool = True,
        log_only_main_category: bool = False,
    ):
        """

        Args:
            embeddings_key: Key to take the embeddings from the batches
            labels_key: Key to take the labels from the batches
            is_query_key: Key to take the information whether every batch sample belongs to the query
            is_gallery_key: Key to take the information whether every batch sample belongs to the gallery
            extra_keys: Keys to accumulate some additional information from the batches
            cmc_top_k: Values of ``k`` to compute ``CMC@k`` metrics
            precision_top_k: Values of ``k`` to compute ``Precision@k`` metrics
            map_top_k: Values of ``k`` to compute ``MAP@k`` metrics
            categories_key: Key to take the samples' categories from the batches (if you have ones)
            postprocessor: Postprocessor which applies some techniques like query reranking
            check_dataset_validity: Set ``True`` if you want to check if all the queries have valid answers in the gallery set

        """
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.extra_keys = extra_keys
        self.cmc_top_k = cmc_top_k
        self.precision_top_k = precision_top_k
        self.map_top_k = map_top_k

        self.categories_key = categories_key
        self.postprocessor = postprocessor

        self.distance_matrix = None
        self.mask_gt = None
        self.metrics = None
        self.mask_to_ignore = None

        self.check_dataset_validity = check_dataset_validity
        self.log_only_main_category = log_only_main_category

        self.visualization_metrics_to_ignore = visualization_metrics_to_ignore

        self.keys_to_accumulate = [self.embeddings_key, self.is_query_key, self.is_gallery_key, self.labels_key]
        if self.categories_key:
            self.keys_to_accumulate.append(self.categories_key)
        if self.extra_keys:
            self.keys_to_accumulate.extend(list(extra_keys))

        self.acc = Accumulator(keys_to_accumulate=self.keys_to_accumulate)

    def setup(self, num_samples: int) -> None:  # type: ignore
        self.distance_matrix = None
        self.mask_gt = None
        self.metrics = None
        self.mask_to_ignore = None

        self.acc.refresh(num_samples=num_samples)

    def update_data(self, data_dict: Dict[str, Any]) -> None:  # type: ignore
        self.acc.update_data(data_dict=data_dict)

    def _calc_matrices(self) -> None:
        embeddings = self.acc.storage[self.embeddings_key]
        labels = self.acc.storage[self.labels_key]
        is_query = self.acc.storage[self.is_query_key]
        is_gallery = self.acc.storage[self.is_gallery_key]

        if self.postprocessor:
            # we have no this functionality yet
            self.postprocessor.process()

        # Note, in some of the datasets part of the samples may appear in both query & gallery.
        # Here we handle this case to avoid picking an item itself as the nearest neighbour for itself
        self.mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)
        self.mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)
        self.distance_matrix = calc_distance_matrix(embeddings=embeddings, is_query=is_query, is_gallery=is_gallery)

    def compute_metrics(self) -> TMetricsDict_ByLabels:  # type: ignore
        if not self.acc.is_storage_full():
            raise ValueError(
                f"Metrics have to be calculated on fully collected data. "
                f"The size of the current storage is less than num samples: "
                f"we've collected {self.acc.collected_samples} out of {self.acc.num_samples}."
            )

        self._calc_matrices()

        args = {
            "cmc_top_k": self.cmc_top_k,
            "precision_top_k": self.precision_top_k,
            "map_top_k": self.map_top_k,
        }

        metrics: TMetricsDict_ByLabels = dict()

        # note, here we do micro averaging
        metrics[self.overall_categories_key] = calc_retrieval_metrics(
            distances=self.distance_matrix,
            mask_gt=self.mask_gt,
            mask_to_ignore=self.mask_to_ignore,
            check_dataset_validity=self.check_dataset_validity,
            reduce=False,
            **args,  # type: ignore
        )

        if self.categories_key is not None:
            categories = np.array(self.acc.storage[self.categories_key])
            is_query = self.acc.storage[self.is_query_key]
            query_categories = categories[is_query]

            for category in np.unique(query_categories):
                mask = query_categories == category

                metrics[category] = calc_retrieval_metrics(
                    distances=self.distance_matrix[mask],  # type: ignore
                    mask_gt=self.mask_gt[mask],  # type: ignore
                    mask_to_ignore=self.mask_to_ignore[mask],  # type: ignore
                    reduce=False,
                    **args,  # type: ignore
                )

        self.metrics_unreduced = metrics
        self.metrics = reduce_metrics(metrics)  # type: ignore

        return self.metrics

    def visualize(self, *args: Any, **kwargs: Any) -> Tuple[Collection[plt.Figure], Collection[str]]:
        metrics_flat = flatten_dict(self.metrics, ignored_keys=self.visualization_metrics_to_ignore)
        figures = []
        titles = []
        for metric_name in metrics_flat:
            if self.log_only_main_category and not metric_name.startswith(OVERALL_CATEGORIES_KEY):
                continue
            fig = self.get_plot_for_worst_queries(
                metric_name=metric_name, topk_queries=LOG_TOPK_ROWS_PER_METRIC, topk_instances=LOG_TOPK_IMAGES_PER_ROW
            )
            log_str = f"top {LOG_TOPK_ROWS_PER_METRIC} worst by {metric_name}"
            figures.append(fig)
            titles.append(log_str)
        return figures, titles

    def ready_to_visualize(self) -> bool:
        return PATHS_KEY in self.extra_keys

    def get_worst_queries_ids(self, metric_name: str, topk: int) -> List[int]:
        metric_values = flatten_dict(self.metrics_unreduced)[metric_name]  # type: ignore
        return torch.topk(metric_values, min(topk, len(metric_values)))[1].tolist()

    def get_plot_for_worst_queries(
        self, metric_name: str, topk_queries: int, topk_instances: int, verbose: bool = False
    ) -> plt.Figure:
        query_ids = self.get_worst_queries_ids(metric_name=metric_name, topk=topk_queries)
        return self.get_plot_for_queries(query_ids=query_ids, top_k=topk_instances, verbose=verbose)

    def get_plot_for_queries(
        self,
        query_ids: List[int],
        top_k: int,
        verbose: bool = True,
    ) -> plt.Figure:
        """
        Visualize the predictions for the query with the indicies <query_ids>.

        Args:
            query_ids: Index of the query
            top_k: Amount of the predictions to show
            verbose: wether to show image paths or not

        """
        assert self.metrics is not None, "We are not ready to plot, because metrics were not calculated yet."

        dist_matrix_with_inf = torch.clone(self.distance_matrix)
        dist_matrix_with_inf[self.mask_to_ignore] = float("inf")

        is_query = self.acc.storage[self.is_query_key]
        is_gallery = self.acc.storage[self.is_gallery_key]

        query_paths = np.array(self.acc.storage[PATHS_KEY])[is_query]
        gallery_paths = np.array(self.acc.storage[PATHS_KEY])[is_gallery]

        fake_coord = np.array([float("nan")] * len(is_query))
        bboxes = list(
            zip(
                self.acc.storage.get(X1_KEY, fake_coord),
                self.acc.storage.get(Y1_KEY, fake_coord),
                self.acc.storage.get(X2_KEY, fake_coord),
                self.acc.storage.get(Y2_KEY, fake_coord),
            )
        )

        query_bboxes = torch.tensor(bboxes)[is_query]
        gallery_bboxes = torch.tensor(bboxes)[is_gallery]

        fig = plt.figure(figsize=(30, 30 / (top_k + 2 + 1) * len(query_ids)))
        for j, query_idx in enumerate(query_ids):
            ids = torch.argsort(dist_matrix_with_inf[query_idx])[:top_k]

            n_gt = self.mask_gt[query_idx].sum()  # type: ignore

            plt.subplot(
                len(query_ids),
                top_k + 1 + N_GT_SHOW_EMBEDDING_METRICS,
                j * (top_k + 1 + N_GT_SHOW_EMBEDDING_METRICS) + 1,
            )

            img = get_img_with_bbox(query_paths[query_idx], query_bboxes[query_idx], BLUE)
            img = square_pad(img)
            if verbose:
                print("Q  ", query_paths[query_idx])
            plt.imshow(img)
            plt.title(f"Query, #gt = {n_gt}")
            plt.axis("off")

            for i, idx in enumerate(ids):
                color = GREEN if self.mask_gt[query_idx, idx] else RED  # type: ignore
                if verbose:
                    print("G", i, gallery_paths[idx])
                plt.subplot(
                    len(query_ids),
                    top_k + N_GT_SHOW_EMBEDDING_METRICS + 1,
                    j * (top_k + 1 + N_GT_SHOW_EMBEDDING_METRICS) + i + 2,
                )
                img = get_img_with_bbox(gallery_paths[idx], gallery_bboxes[idx], color)
                img = square_pad(img)
                plt.title(f"{i} - {round(dist_matrix_with_inf[query_idx, idx].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            gt_ids = self.mask_gt[query_idx].nonzero(as_tuple=True)[0][:N_GT_SHOW_EMBEDDING_METRICS]  # type: ignore

            for i, gt_idx in enumerate(gt_ids):
                plt.subplot(
                    len(query_ids),
                    top_k + N_GT_SHOW_EMBEDDING_METRICS + 1,
                    j * (top_k + 1 + N_GT_SHOW_EMBEDDING_METRICS) + i + top_k + 2,
                )
                img = get_img_with_bbox(gallery_paths[gt_idx], gallery_bboxes[gt_idx], GRAY)
                img = square_pad(img)
                plt.title("GT " + str(round(dist_matrix_with_inf[query_idx, gt_idx].item(), 3)))
                plt.imshow(img)
                plt.axis("off")

        fig.tight_layout()
        return fig


class EmbeddingMetricsDDP(EmbeddingMetrics, IBasicMetricDDPWithVisualization):
    def sync(self) -> None:
        self.acc = self.acc.sync()


__all__ = ["TMetricsDict_ByLabels", "EmbeddingMetrics", "EmbeddingMetricsDDP"]
