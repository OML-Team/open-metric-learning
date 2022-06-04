from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from torch import nonzero

from oml.const import T_Str2Int_or_Int2Str
from oml.functional.metrics import (
    TMetricsDict,
    calc_distance_matrix,
    calc_gt_mask,
    calc_retrieval_metrics,
)
from oml.interfaces.metrics import IBasicMetric
from oml.metrics.accumulation import Accumulator

TMetricsDict_ByLabels = Dict[Union[str, int], TMetricsDict]


class IPostprocessor:
    def process(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError


class EmbeddingMetrics(IBasicMetric):
    metric_name = ""

    def __init__(
        self,
        embeddings_key: str = "embeddings",
        labels_key: str = "labels",
        is_query_key: str = "is_query",
        is_gallery_key: str = "is_gallery",
        extra_keys: Tuple[str, ...] = (),
        top_k: Tuple[int, ...] = (1,),
        need_cmc: bool = True,
        need_precision: bool = False,
        need_map: bool = False,
        categories_key: Optional[str] = None,
        categories_names_mapping: Optional[T_Str2Int_or_Int2Str] = None,
        postprocessor: Optional[IPostprocessor] = None,
    ):
        if (categories_names_mapping is not None) and (categories_key is None):
            raise ValueError(
                "You have not specified category key but specified the mapping for " "the categories in the same time."
            )

        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.extra_keys = extra_keys
        self.top_k = top_k
        self.need_cmc = need_cmc
        self.need_precision = need_precision
        self.need_map = need_map
        self.categories_key = categories_key
        self.categories_names_mapping = categories_names_mapping
        self.postprocessor = postprocessor

        self.distance_matrix = None
        self.mask_gt = None
        self.metrics = None
        self.mask_to_ignore = None

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
            embeddings = self.postprocessor.process(embeddings, labels, is_query, is_gallery)

        self.mask_gt = calc_gt_mask(labels=labels, is_query=is_query, is_gallery=is_gallery)
        self.distance_matrix = calc_distance_matrix(embeddings=embeddings, is_query=is_query, is_gallery=is_gallery)

        # Note, in some of the datasets part of the samples may appear in both query & gallery.
        # Here we handle this case to avoid picking an item itself as the nearest neighbour for itself
        ids_query = nonzero(is_query).squeeze()
        ids_gallery = nonzero(is_gallery).squeeze()
        self.mask_to_ignore = ids_query[..., None] == ids_gallery[None, ...]

    def compute_metrics(self) -> TMetricsDict_ByLabels:  # type: ignore
        if not self.acc.is_storage_full():
            raise ValueError(
                f"Metrics have to be calculated on fully collected data. "
                f"The size of the current storage is less than num samples: "
                f"we've collected {self.acc.collected_samples} out of {self.acc.num_samples}."
            )

        self._calc_matrices()

        args = {
            "top_k": self.top_k,
            "need_cmc": self.need_cmc,
            "need_map": self.need_map,
            "need_precision": self.need_precision,
        }

        metrics: TMetricsDict_ByLabels = dict()

        # note, here we do micro averaging
        metrics["OVERALL"] = calc_retrieval_metrics(
            distances=self.distance_matrix,
            mask_gt=self.mask_gt,
            mask_to_ignore=self.mask_to_ignore,
            **args,  # type: ignore
        )

        if self.categories_key is not None:
            categories = np.array(self.acc.storage[self.categories_key])
            is_query = self.acc.storage[self.is_query_key]
            query_categories = categories[is_query]

            for category in np.unique(query_categories):
                mask = query_categories == category

                if self.categories_names_mapping is not None:
                    category = self.categories_names_mapping[category]

                metrics[category] = calc_retrieval_metrics(
                    distances=self.distance_matrix[mask],  # type: ignore
                    mask_gt=self.mask_gt[mask],  # type: ignore
                    mask_to_ignore=self.mask_to_ignore[mask],  # type: ignore
                    **args,  # type: ignore
                )

        self.metrics = metrics  # type: ignore

        return metrics

    @property
    def get_metrics(self) -> TMetricsDict:
        if self.metrics is None:
            raise ValueError(
                f"Metrics have not been calculated." f"Make sure you've called {self.compute_metrics.__name__}"
            )

        return self.metrics
