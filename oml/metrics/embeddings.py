from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from oml.functional.metrics import (
    TMetricsDict,
    calc_distance_matrix,
    calc_gt_mask,
    calc_mask_to_ignore,
    calc_retrieval_metrics,
)
from oml.interfaces.metrics import IBasicMetric
from oml.interfaces.post_processor import IPostprocessor
from oml.metrics.accumulation import Accumulator

TMetricsDict_ByLabels = Dict[Union[str, int], TMetricsDict]


class EmbeddingMetrics(IBasicMetric):
    metric_name = ""

    def __init__(
        self,
        embeddings_key: str = "embeddings",
        labels_key: str = "labels",
        is_query_key: str = "is_query",
        is_gallery_key: str = "is_gallery",
        extra_keys: Tuple[str, ...] = (),
        cmc_top_k: Tuple[int, ...] = (5,),
        precision_top_k: Tuple[int, ...] = (5,),
        map_top_k: Tuple[int, ...] = (5,),
        categories_key: Optional[str] = None,
        postprocessor: Optional[IPostprocessor] = None,
        check_dataset_validity: bool = True,
    ):
        """
        This class accumulates the information from the batches and embeddings produced by the model
        at every batch in epoch. After all of the samples have been stored you can call the function
        which computes retrievals metrics. To get the needed information from the batches it uses
        keys which have to be provided as init arguments. Please, check the example in the <Using pure PyTorch>
        section in the main Readme.md.

        Args:
            embeddings_key: Key to take the embeddings from the batches
            labels_key: Key to take the labels from the batches
            is_query_key: Key to take the information whether every batch sample belongs to the query
            is_gallery_key: Key to take the information whether every batch sample belongs to the gallery
            extra_keys: Keys to accumulate some additional information from the batches
            cmc_top_k: Values of k to compute CMC@k metrics
            precision_top_k: Values of k to compute Precision@k metrics
            map_top_k: Values of k to compute MAP@k metrics
            categories_key: Key to take the samples categories from the batches (if you have ones)
            postprocessor: IPostprocessor which applies some techniques like query reranking
            check_dataset_validity: check if all queries have their galleries

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
        metrics[self.main_category_key] = calc_retrieval_metrics(
            distances=self.distance_matrix,
            mask_gt=self.mask_gt,
            mask_to_ignore=self.mask_to_ignore,
            check_dataset_validity=self.check_dataset_validity,
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
                    **args,  # type: ignore
                )

        self.metrics = metrics  # type: ignore

        return metrics


__all__ = ["TMetricsDict_ByLabels", "EmbeddingMetrics"]
