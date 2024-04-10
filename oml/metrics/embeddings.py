from copy import deepcopy
from copy import deepcopy
from pprint import pprint
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor

from oml.const import (
    BLACK,
    BLUE,
    EMBEDDINGS_KEY,
    GRAY,
    GREEN,
    IS_GALLERY_COLUMN,
    IS_GALLERY_KEY,
    IS_QUERY_COLUMN,
    IS_QUERY_KEY,
    LABELS_KEY,
    LOG_TOPK_IMAGES_PER_ROW,
    LOG_TOPK_ROWS_PER_METRIC,
    N_GT_SHOW_EMBEDDING_METRICS,
    OVERALL_CATEGORIES_KEY,
    RED,
)
from oml.datasets.base import DatasetQueryGallery
from oml.ddp.utils import is_main_process
from oml.functional.metrics import (
    TMetricsDict,
    calc_gt_mask,
    calc_retrieval_metrics,
    calc_topological_metrics,
    reduce_metrics,
)
from oml.interfaces.metrics import IMetricDDP, IMetricVisualisable
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.metrics.accumulation import Accumulator
from oml.utils.misc import flatten_dict
from oml.utils.misc_torch import pairwise_dist, take_2d

TMetricsDict_ByLabels = Dict[Union[str, int], TMetricsDict]


def batched_knn(
    embeddings: FloatTensor,
    ids_query: LongTensor,
    ids_gallery: LongTensor,
    k_neigh: int,
    bs: int,
    ignoring_groups: Optional[BoolTensor] = None,
) -> Tuple[FloatTensor, LongTensor, BoolTensor]:
    assert embeddings.shape == (len(ids_query), len(ids_gallery))

    emb_q = embeddings[ids_query]
    emb_g = embeddings[ids_gallery]

    nq = len(emb_q)

    ids_neigh = LongTensor(nq, k_neigh)
    distances_neigh = FloatTensor(nq, k_neigh)
    mask_to_ignore_neigh = BoolTensor(nq, k_neigh)

    for i in range(0, nq, bs):
        distances_batch = pairwise_dist(x1=emb_q[i : i + bs], x2=emb_g)

        # todo: separate function mask_to_ignore
        mask_to_ignore_batch = ids_query[i : i + bs][..., None] == ids_gallery[None, ...]
        if ignoring_groups is not None:
            mask_groups = ignoring_groups[ids_query][i : i + bs][..., None] == ignoring_groups[ids_gallery][None, ...]
            mask_to_ignore_batch = np.logical_or(mask_to_ignore_batch, mask_groups).bool()

        # todo: apply
        distances_batch[mask_to_ignore_batch] = float("inf")

        distances_batch_neigh, ids_batch_neigh = torch.topk(distances_batch, k=k_neigh, largest=False, sorted=True)

        ids_neigh[i : i + bs] = ids_batch_neigh
        distances_neigh[i : i + bs] = distances_batch_neigh
        mask_to_ignore_neigh[i : i + bs] = take_2d(mask_to_ignore_batch, ids_batch_neigh)

    return distances_neigh, ids_neigh, mask_to_ignore_neigh


def test_batched_knn(n_generations: int = 5) -> None:
    for _ in range(n_generations):
        nq, ng, dim = 30, 50, 16
        k, bs = 13, 9

        query = torch.randn(nq, dim).float()
        gallery = torch.randn(ng, dim).float()
        distances_neigh, ids_neih = batched_knn(query, gallery, k_neigh=k, batch_size=bs)

        distances = pairwise_dist(query, gallery)
        distances_top_k_expected, ids_top_k_expected = torch.topk(distances, k=k, largest=False, sorted=True)

        assert torch.isclose(distances_neigh, distances_top_k_expected).all()
        assert (ids_neih == ids_top_k_expected).all()


class Prediction:
    def __init__(
        self,
        distances: FloatTensor,
        gallery_ids: LongTensor,
        mask_gt: Optional[BoolTensor] = None,
        n_gts: Optional[BoolTensor] = None,
    ):
        assert distances.shape == gallery_ids.shape == mask_gt.shape

        self.distances = distances  # QxK
        self.retrieved_ids = gallery_ids  # QxK
        self.mask_gt = mask_gt  # QxK
        self.n_gts = n_gts  # Qx1 todo

        # todo: self.is_partial_prediction?

    @classmethod
    def compute_from_embeddings(
        cls,
        embeddings: FloatTensor,
        is_query: BoolTensor,
        is_gallery: BoolTensor,
        ignoring_groups: Optional[LongTensor] = None,
        labels_gt: Optional[LongTensor] = None,
        k_top_results: int = 500,
    ):
        assert len(embeddings) == len(is_query) == len(is_gallery)
        assert (ignoring_groups is None) or (len(ignoring_groups) == len(embeddings))
        assert (labels_gt is None) or (len(labels_gt) == len(embeddings))

        distances_top, gallery_ids, mask_to_ignore_top = batched_knn(
            embeddings=embeddings,
            ids_query=is_query.nonzero(),
            ids_gallery=is_gallery.nonzero(),
            ignoring_groups=ignoring_groups,
            k_neigh=k_top_results,
            bs=1000,
        )

        if labels_gt:
            mask_gt = calc_gt_mask(labels=labels_gt, is_query=is_query, is_gallery=is_gallery)
            mask_gt[mask_to_ignore_top] = False
            # todo: deu to need of n_gt, should replace mask_gt -> gt_ids???

            # 2 arguments
            # 1 arg: keep ful mask_gt (incosistency)
            # 1 arg: gt_ids [[2, 3], [1], [3, 5]] - index error (+)

            mask_gt_top = take_2d(mask_gt, gallery_ids)
            n_gts = mask_gt.sum(dim=1)
        else:
            mask_gt_top = None
            n_gts = None

        return Prediction(
            distances=distances_top,
            gallery_ids=gallery_ids,
            mask_gt=mask_gt_top,
            n_gts=n_gts,
        )

    # todo: rename visualise
    def plot_queries(
        self, query_ids: List[int], n_instances: int, dataset: DatasetQueryGallery, verbose: bool = True
    ) -> plt.Figure:
        ii_query = torch.tensor(dataset.df[IS_QUERY_COLUMN]).long()  # todo
        ii_gallery = torch.tensor(dataset.df[IS_GALLERY_COLUMN]).long()  # todo

        n_gt = N_GT_SHOW_EMBEDDING_METRICS if (self.mask_gt is not None) else 0

        fig = plt.figure(figsize=(16, 16 / (n_instances + n_gt + 1) * len(query_ids)))

        n_rows, n_cols = len(query_ids), n_instances + 1 + N_GT_SHOW_EMBEDDING_METRICS

        for j, query_idx in enumerate(query_ids):

            plt.subplot(n_rows, n_cols, j * (n_instances + 1 + n_gt) + 1)

            img = dataset.visualize(ii_query[query_idx], color=BLUE)

            if verbose:
                print("Q  ", dataset[ii_query[query_idx]][dataset.paths_key])

            plt.imshow(img)
            plt.title("Query")
            plt.axis("off")

            ids = self.retrieved_ids[query_idx][:n_instances]

            for i, idx in enumerate(ids):
                if self.mask_gt is not None:
                    color = GREEN if self.mask_gt[query_idx, idx] else RED  # type: ignore
                else:
                    color = BLACK

                if verbose:
                    print("G", i, dataset[ii_gallery[idx]][dataset.paths_key])

                plt.subplot(n_rows, n_cols, j * (n_instances + 1 + n_gt) + i + 2)
                img = dataset.visualize(ii_gallery[idx], color=color)

                plt.title(f"{i} - {round(self.distances[query_idx, idx].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            # todo: it's not correct! the gt_mask is not full
            if self.mask_gt is not None:
                ids = self.mask_gt[query_idx].nonzero(as_tuple=True)[0][:n_gt]  # type: ignore

                for i, idx in enumerate(ids):
                    plt.subplot(n_rows, n_cols, j * (n_instances + 1 + n_gt) + i + n_instances + 2)

                    img = dataset.visualize(ii_gallery[idx], color=GRAY)
                    plt.title("GT " + str(round(self.distances[query_idx, idx].item(), 3)))
                    plt.imshow(img)
                    plt.axis("off")

        fig.tight_layout()
        return fig


def validate_dataset(mask_gt: Tensor, mask_to_ignore: Tensor) -> None:
    # todo we don't have full mask_gt and mask to ignore in one place
    is_valid = (mask_gt & ~mask_to_ignore).any(1).all()

    if not is_valid:
        raise RuntimeError("There are queries without available correct answers in the gallery!")


class EmbeddingMetrics(IMetricVisualisable):
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
        fmr_vals: Tuple[float, ...] = tuple(),
        pcf_variance: Tuple[float, ...] = (0.5,),
        categories_key: Optional[str] = None,
        sequence_key: Optional[str] = None,
        postprocessor: Optional[IDistancesPostprocessor] = None,
        metrics_to_exclude_from_visualization: Iterable[str] = (),
        return_only_overall_category: bool = False,
        visualize_only_overall_category: bool = True,
        verbose: bool = True,
    ):
        """

        Args:
            embeddings_key: Key to take the embeddings from the batches
            labels_key: Key to take the labels from the batches
            is_query_key: Key to take the information whether every batch sample belongs to the query
            is_gallery_key: Key to take the information whether every batch sample belongs to the gallery
            extra_keys: Keys to accumulate some additional information from the batches
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
            categories_key: Key to take the samples' categories from the batches (if you have ones)
            sequence_key: Key to take sequence ids from the batches (if you have ones)
            postprocessor: Postprocessor which applies some techniques like query reranking
            metrics_to_exclude_from_visualization: Names of the metrics to exclude from the visualization. It will not
             affect calculations.
            return_only_overall_category: Set ``True`` if you want to return only the aggregated metrics
            visualize_only_overall_category: Set ``False`` if you want to visualize each category separately
            verbose: Set ``True`` if you want to print metrics

        """
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.extra_keys = extra_keys
        self.cmc_top_k = cmc_top_k
        self.precision_top_k = precision_top_k
        self.map_top_k = map_top_k
        self.fmr_vals = fmr_vals
        self.pcf_variance = pcf_variance

        self.categories_key = categories_key
        self.sequence_key = sequence_key
        self.postprocessor = postprocessor

        self.prediction = None
        self.metrics = None
        self.metrics_unreduced = None

        self.visualize_only_overall_category = visualize_only_overall_category
        self.return_only_overall_category = return_only_overall_category

        self.metrics_to_exclude_from_visualization = ["fnmr@fmr", "pcf", *metrics_to_exclude_from_visualization]
        self.verbose = verbose

        keys_to_accumulate = [self.embeddings_key, self.is_query_key, self.is_gallery_key, self.labels_key]
        if self.categories_key:
            keys_to_accumulate.append(self.categories_key)
        if self.sequence_key:
            keys_to_accumulate.append(self.sequence_key)
        if self.extra_keys:
            keys_to_accumulate.extend(list(extra_keys))
        if self.postprocessor:
            keys_to_accumulate.extend(self.postprocessor.needed_keys)

        self.keys_to_accumulate = tuple(set(keys_to_accumulate))
        self.acc = Accumulator(keys_to_accumulate=self.keys_to_accumulate)

    def setup(self, num_samples: int) -> None:  # type: ignore
        self.prediction = None  # todo: garbage collection?
        self.metrics = None
        self.metrics_unreduced = None

        self.acc.refresh(num_samples=num_samples)

    def update_data(self, data_dict: Dict[str, Any]) -> None:  # type: ignore
        self.acc.update_data(data_dict=data_dict)

    def _compute_prediction(self) -> None:
        sequence_ids = self.acc.storage[self.sequence_key] if self.sequence_key is not None else None

        if isinstance(sequence_ids, list):
            # if sequence ids are strings we get list here
            # todo: check types in other places
            sequence_ids = np.array(sequence_ids)

        self.prediction = Prediction.compute_from_embeddings(
            embeddings=self.acc.storage[self.embeddings_key].float(),
            is_query=self.acc.storage[self.is_query_key].bool(),
            is_gallery=self.acc.storage[self.is_gallery_key].bool(),
            labels_gt=self.acc.storage[self.labels_key].long(),
            ignoring_groups=sequence_ids,
        )

        if self.postprocessor:
            # todo: refactor logic (or just keep full distances matrix for now, but need to check
            # todo: topk performance in the case of big k)
            # topk
            self.prediction = self.postprocessor.process_by_dict(self.prediction.distances, data=self.acc.storage)
            # reverse

    def compute_metrics(self) -> TMetricsDict_ByLabels:  # type: ignore
        if not self.acc.is_storage_full():
            raise ValueError(
                f"Metrics have to be calculated on fully collected data. "
                f"The size of the current storage is less than num samples: "
                f"we've collected {self.acc.collected_samples} out of {self.acc.num_samples}."
            )

        self._compute_prediction()

        metrics_dict: TMetricsDict_ByLabels = dict()

        # dist_topk = ...

        # note, here we do micro averaging
        metrics_unreduced = calc_retrieval_metrics(
            distances=self.prediction.distance_matrix,
            mask_gt=self.prediction.mask_gt,
            reduce=False,
            mask_to_ignore=None,  # we already applied it
            cmc_top_k=self.cmc_top_k,
            precision_top_k=self.precision_top_k,
            map_top_k=self.map_top_k,
            fmr_vals=self.fmr_vals,  # todo: check types
        )
        metrics_dict[self.overall_categories_key] = metrics_unreduced

        embeddings = self.acc.storage[self.embeddings_key]
        metrics_dict[self.overall_categories_key].update(
            calc_topological_metrics(embeddings, pcf_variance=self.pcf_variance)
        )

        if self.categories_key is not None:
            categories = np.array(self.acc.storage[self.categories_key])  # n
            is_query = self.acc.storage[self.is_query_key]  # n
            query_categories = categories[is_query]  # nq

            for category in np.unique(query_categories):
                mask = query_categories == category
                metrics_dict[category] = reduce_metrics(metrics_unreduced[mask])  # todo: proces dict

                mask = categories == category
                metrics_dict[category].update(
                    calc_topological_metrics(embeddings[mask], pcf_variance=self.pcf_variance)
                )

        self.metrics_unreduced = metrics_unreduced  # type: ignore
        self.metrics = reduce_metrics(metrics_dict)  # type: ignore

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

    def visualize(self, dataset: DatasetQueryGallery) -> Tuple[Collection[plt.Figure], Collection[str]]:
        """
        Visualize worst queries by all the available metrics.
        # todo: signature
        """
        metrics_flat = flatten_dict(self.metrics, ignored_keys=self.metrics_to_exclude_from_visualization)
        figures = []
        titles = []
        for metric_name in metrics_flat:
            if self.visualize_only_overall_category and not metric_name.startswith(OVERALL_CATEGORIES_KEY):
                continue
            fig = self.get_plot_for_worst_queries(
                metric_name=metric_name,
                n_queries=LOG_TOPK_ROWS_PER_METRIC,
                n_instances=LOG_TOPK_IMAGES_PER_ROW,
                data=dataset,
            )
            log_str = f"top {LOG_TOPK_ROWS_PER_METRIC} worst by {metric_name}".replace("/", "_")
            figures.append(fig)
            titles.append(log_str)
        return figures, titles

    def get_worst_queries_ids(self, metric_name: str, n_queries: int) -> List[int]:
        metric_values = flatten_dict(self.metrics_unreduced)[metric_name]  # type: ignore
        return torch.topk(metric_values, min(n_queries, len(metric_values)), largest=False)[1].tolist()

    def get_plot_for_worst_queries(
        self, metric_name: str, n_queries: int, n_instances: int, dataset: DatasetQueryGallery, verbose: bool = False
    ) -> plt.Figure:
        query_ids = self.get_worst_queries_ids(metric_name=metric_name, n_queries=n_queries)
        return self.get_plot_for_queries(query_ids=query_ids, n_instances=n_instances, dataset=dataset, verbose=verbose)

    def get_plot_for_queries(
        self, query_ids: List[int], n_instances: int, dataset: DatasetQueryGallery, verbose: bool = True
    ) -> plt.Figure:
        """
        Visualize the predictions for the query with the indicies <query_ids>.

        Args:
            query_ids: Index of the query
            n_instances: Amount of the predictions to show
            dataset: todo
            verbose: wether to show image paths or not

        """
        return self.prediction.plot_queries(query_ids, n_instances=n_instances, verbose=verbose, dataset=dataset)


class EmbeddingMetricsDDP(EmbeddingMetrics, IMetricDDP):
    def sync(self) -> None:
        self.acc = self.acc.sync()


__all__ = ["TMetricsDict_ByLabels", "EmbeddingMetrics", "EmbeddingMetricsDDP"]
