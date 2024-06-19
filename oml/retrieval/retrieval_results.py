import warnings
from copy import deepcopy
from pprint import pformat
from typing import Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import FloatTensor, LongTensor

from oml.const import (
    BLACK,
    BLUE,
    GRAY,
    GREEN,
    N_GT_SHOW_EMBEDDING_METRICS,
    RED,
    SEQUENCE_COLUMN,
    TColor,
)
from oml.functional.knn import batched_knn, batched_knn_qg
from oml.interfaces.datasets import (
    IBaseDataset,
    ILabeledDataset,
    IQueryGalleryDataset,
    IQueryGalleryLabeledDataset,
    IVisualizableDataset,
)
from oml.utils.misc_torch import is_sorted_tensor


def get_sequence_from_dataset(dataset: IBaseDataset) -> Optional[LongTensor]:
    if SEQUENCE_COLUMN in dataset.extra_data:
        sequence = pd.Series(dataset.extra_data[SEQUENCE_COLUMN])
        sequence_ids = LongTensor(pd.factorize(sequence, sort=True)[0])
    else:
        sequence_ids = None

    return sequence_ids


class RetrievalResults:
    _max_elements_in_str_repr: int = 100

    def __init__(
        self,
        distances: Sequence[FloatTensor],
        retrieved_ids: Sequence[LongTensor],
        gt_ids: Optional[Sequence[LongTensor]] = None,
    ):
        """
        Args:
            distances: Sorted distances from queries to the first gallery items with the size of ``n_query``.
            retrieved_ids: First gallery indices retrieved for every query with the size of ``n_query``.
                Every index is within the range ``(0, n_gallery - 1)``.
            gt_ids: Gallery indices relevant to every query with the size of ``n_query``.
                Every element is within the range ``(0, n_gallery - 1)``

        """
        for d, r in zip(distances, retrieved_ids):
            if not is_sorted_tensor(d):
                raise RuntimeError(f"Distances must be sorted: {d}.")
            if not len(torch.unique(r[:100])) == len(r[:100]):  # it's too expensive to check all the ids!
                raise RuntimeError("Retrieved ids must be unique.")
            if not len(d) == len(r):
                raise RuntimeError("Number of distances and retrieved items must match.")
            if (d.ndim != 1) or (r.ndim != 1):
                raise RuntimeError("Distances and retrieved items must be 1-d tensors.")

        if gt_ids is not None:
            assert len(distances) == len(gt_ids)
            if any(len(x) == 0 for x in gt_ids):
                warnings.warn("Some of the queries don't have available gts.")

        self._distances = tuple(distances)
        self._retrieved_ids = tuple(retrieved_ids)
        self._gt_ids = tuple(gt_ids) if gt_ids is not None else None

    @property
    def distances(self) -> Tuple[FloatTensor, ...]:
        """
        Returns:
            Sorted distances from queries to the first gallery items with the size of ``n_query``.
        """
        return self._distances

    @property
    def retrieved_ids(self) -> Tuple[LongTensor, ...]:
        """
        Returns:
            First gallery indices retrieved for every query with the size of ``n_query``.
            Every index is within the range ``(0, n_gallery - 1)``.
        """
        return self._retrieved_ids

    @property
    def gt_ids(self) -> Optional[Tuple[LongTensor, ...]]:
        """
        Returns:
            Gallery indices relevant to every query with the size of ``n_query``.
            Every element is within the range ``(0, n_gallery - 1)``
        """
        return self._gt_ids

    @property
    def n_retrieved_items(self) -> int:
        """
        Returns:
            Number of items retrieved for each query. If queries have different number of retrieved items,
            returns the maximum of them.

        """
        return max(len(x) for x in self.retrieved_ids)

    def is_empty(self) -> bool:
        return all(len(rids) == 0 for rids in self.retrieved_ids)

    def deepcopy(self) -> "RetrievalResults":
        return RetrievalResults(
            retrieved_ids=deepcopy(self.retrieved_ids), distances=deepcopy(self.distances), gt_ids=deepcopy(self.gt_ids)
        )

    @classmethod
    def from_embeddings(
        cls,
        embeddings: FloatTensor,
        dataset: Union[IQueryGalleryDataset, IQueryGalleryLabeledDataset],
        n_items: int = 100,
        verbose: bool = False,
    ) -> "RetrievalResults":
        """
        Args:
            embeddings: The result of inference with the shape of ``[dataset_len, emb_dim]``.
            dataset: Dataset having query/gallery split.
            n_items: Number of the closest gallery items to retrieve. It may be clipped by
                gallery size if needed. Note, some queries may get less than this number of retrieved items if they
                don't have enough gallery items available.
            verbose: Set ``True`` to see progress bar.

        """
        assert len(embeddings) == len(dataset), "Embeddings and dataset must have the same size."

        sequence_ids = get_sequence_from_dataset(dataset)

        labels_gt = dataset.get_labels() if isinstance(dataset, IQueryGalleryLabeledDataset) else None

        distances, retrieved_ids, gt_ids = batched_knn(
            embeddings=embeddings,
            ids_query=dataset.get_query_ids(),
            ids_gallery=dataset.get_gallery_ids(),
            labels_gt=labels_gt,
            sequence_ids=sequence_ids,
            top_n=n_items,
            verbose=verbose,
        )

        return RetrievalResults(distances=distances, retrieved_ids=retrieved_ids, gt_ids=gt_ids)

    @classmethod
    def from_embeddings_qg(
        cls,
        embeddings_query: FloatTensor,
        embeddings_gallery: FloatTensor,
        dataset_query: Union[IBaseDataset, ILabeledDataset],
        dataset_gallery: Union[IBaseDataset, ILabeledDataset],
        n_items: int = 100,
        verbose: bool = False,
    ) -> "RetrievalResults":
        """
        Args:
            embeddings_query: The result of inference with the shape of ``[n_queries, emb_dim]``.
            embeddings_gallery: The result of inference with the shape of ``[n_galleries, emb_dim]``.
            dataset_query: Dataset of queries with the length of ``n_queries``.
            dataset_gallery: Dataset of galleries with the length of ``n_galleries``.
            n_items: Number of the closest gallery items to retrieve. It may be clipped by
                gallery size if needed. Note, some queries may get less than this number of retrieved items if they
                don't have enough gallery items available.
            verbose: Set ``True`` to see progress bar.

        """
        assert len(embeddings_query) == len(dataset_query), "Embeddings and dataset must have the same size."
        assert len(embeddings_gallery) == len(dataset_gallery), "Embeddings and dataset must have the same size."

        labels_query = dataset_query.get_labels() if isinstance(dataset_query, ILabeledDataset) else None
        labels_gallery = dataset_gallery.get_labels() if isinstance(dataset_gallery, ILabeledDataset) else None

        sequence_ids_query = get_sequence_from_dataset(dataset_query)
        sequence_ids_gallery = get_sequence_from_dataset(dataset_gallery)

        distances, retrieved_ids, gt_ids = batched_knn_qg(
            embeddings_query=embeddings_query,
            embeddings_gallery=embeddings_gallery,
            ids_query=None,
            ids_gallery=None,
            labels_query=labels_query,
            labels_gallery=labels_gallery,
            sequence_ids_query=sequence_ids_query,
            sequence_ids_gallery=sequence_ids_gallery,
            top_n=n_items,
            verbose=verbose,
        )

        return RetrievalResults(distances=distances, retrieved_ids=retrieved_ids, gt_ids=gt_ids)

    def __str__(self) -> str:
        m = self._max_elements_in_str_repr

        txt = (
            f"Maximum number of retrieved items: {self.n_retrieved_items}.\n"
            f"Distances to the retrieved items:\n{pformat(self.distances[:m])}.\n"
            f"Ids of the retrieved gallery items:\n{pformat(self.retrieved_ids[:m])}.\n"
        )

        if self.gt_ids is None:
            txt += "Ground truths are unknown.\n"
        else:
            gt_ids_list = [x.tolist() for x in self.gt_ids]
            txt += f"Ground truth gallery ids are:\n{pformat(gt_ids_list[:m])}.\n"

        return txt

    def visualize_qg(
        self,
        query_ids: List[int],
        dataset_query: IVisualizableDataset,
        dataset_gallery: IVisualizableDataset,
        n_galleries_to_show: int = 5,
        n_gt_to_show: int = N_GT_SHOW_EMBEDDING_METRICS,
        verbose: bool = False,
        show: bool = False,
    ) -> plt.Figure:
        """
        Args:
            query_ids: Query indices within the range of ``(0, n_query - 1)``.
            dataset_query: Dataset of queries supporting visualisation, with the length of ``n_query``.
            dataset_gallery: Dataset of queries supporting visualisation, with the length of ``n_gallery``.
            n_galleries_to_show: Number of closest gallery items to show.
            n_gt_to_show: Number of ground truth gallery items to show for reference (if available).
            verbose: Set ``True`` to allow prints.
            show: Set ``True`` to instantly visualise the resulted figure.

        """
        dq_name = dataset_query.__class__.__name__
        dg_name = dataset_gallery.__class__.__name__

        if not isinstance(dataset_query, IVisualizableDataset):
            raise TypeError(f"Query dataset has to support {IVisualizableDataset.__name__}. Got {dq_name}.")

        if not isinstance(dataset_gallery, IVisualizableDataset):
            raise TypeError(f"Gallery dataset has to support {IVisualizableDataset.__name__}. Got {dg_name}.")

        nq1, nq2 = len(self.retrieved_ids), len(dataset_query)
        if nq1 != nq2:
            raise RuntimeError(
                f"Number of queries in {self.__class__.__name__} and {dq_name} must match: {nq1} != {nq2}"
            )

        if verbose:
            print(f"Visualizing {n_galleries_to_show} for the following query ids: {query_ids}.")

        def visualize_query_fn(item: int, color: TColor) -> np.ndarray:
            return dataset_query.visualize(item=item, color=color)

        def visualize_gallery_fn(item: int, color: TColor) -> np.ndarray:
            return dataset_gallery.visualize(item=item, color=color)

        return self.visualize_with_functions(
            query_ids=query_ids,
            visualize_query_fn=visualize_query_fn,
            visualize_gallery_fn=visualize_gallery_fn,
            n_galleries_to_show=n_galleries_to_show,
            n_gt_to_show=n_gt_to_show,
            show=show,
        )

    def visualize(
        self,
        query_ids: List[int],
        dataset: IQueryGalleryDataset,
        n_galleries_to_show: int = 5,
        n_gt_to_show: int = N_GT_SHOW_EMBEDDING_METRICS,
        verbose: bool = False,
        show: bool = False,
    ) -> plt.Figure:
        """
        Args:
            query_ids: Query indices within the range of ``(0, n_query - 1)``.
            dataset: Dataset that provides query-gallery split and supports visualisation.
            n_galleries_to_show: Number of closest gallery items to show.
            n_gt_to_show: Number of ground truth gallery items to show for reference (if available).
            verbose: Set ``True`` to allow prints.
            show: Set ``True`` to instantly visualise the resulted figure.

        """
        dataset_name = dataset.__class__.__name__
        if not isinstance(dataset, IVisualizableDataset):
            raise TypeError(f"Dataset has to support {IVisualizableDataset.__name__}. Got {dataset_name}.")
        if not isinstance(dataset, IQueryGalleryDataset):
            raise TypeError(f"Dataset has to support {IQueryGalleryDataset.__name__}. Got {dataset_name}.")

        nq1, nq2 = len(self.retrieved_ids), len(dataset.get_query_ids())
        if nq1 != nq2:
            raise RuntimeError(
                f"Number of queries in {self.__class__.__name__} and {dataset_name} " f"must match: {nq1} != {nq2}"
            )

        if verbose:
            print(f"Visualizing {n_galleries_to_show} for the following query ids: {query_ids}.")

        def visualize_query_fn(item: int, color: TColor) -> np.ndarray:
            return dataset.visualize(item=dataset.get_query_ids()[item].item(), color=color)

        def visualize_gallery_fn(item: int, color: TColor) -> np.ndarray:
            return dataset.visualize(item=dataset.get_gallery_ids()[item].item(), color=color)

        return self.visualize_with_functions(
            query_ids=query_ids,
            visualize_query_fn=visualize_query_fn,
            visualize_gallery_fn=visualize_gallery_fn,
            n_galleries_to_show=n_galleries_to_show,
            n_gt_to_show=n_gt_to_show,
            show=show,
        )

    def visualize_with_functions(
        self,
        query_ids: List[int],
        visualize_query_fn: Callable[[int, TColor], np.ndarray],
        visualize_gallery_fn: Callable[[int, TColor], np.ndarray],
        n_galleries_to_show: int = 5,
        n_gt_to_show: int = N_GT_SHOW_EMBEDDING_METRICS,
        show: bool = False,
    ) -> plt.Figure:
        """
        Args:
            query_ids: Query indices within the range of ``(0, n_query - 1)``.
            visualize_query_fn: Function plotting ``i-th`` query with respect to the given color.
            visualize_gallery_fn: Function plotting ``j-th`` gallery with respect to the given color.
            n_galleries_to_show: Number of closest gallery items to show.
            n_gt_to_show: Number of ground truth gallery items to show for reference (if available).
            show: Set ``True`` to instantly visualize the resulted figure.

        """

        max_presented_galleries = max(len(self.retrieved_ids[iq]) for iq in query_ids)
        n_galleries_to_show = min(n_galleries_to_show, max_presented_galleries)
        n_gt_to_show = n_gt_to_show if (self.gt_ids is not None) else 0

        fig = plt.figure(figsize=(16, 16 / (n_galleries_to_show + n_gt_to_show + 1) * len(query_ids)))
        n_rows, n_cols = len(query_ids), n_galleries_to_show + 1 + n_gt_to_show

        # iterate over queries
        for i, query_idx in enumerate(query_ids):

            plt.subplot(n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + 1)

            img = visualize_query_fn(query_idx, BLUE)

            plt.imshow(img)
            plt.title(f"Query #{query_idx}")
            plt.axis("off")

            # iterate over retrieved items
            for j, ret_idx in enumerate(self.retrieved_ids[query_idx][:n_galleries_to_show]):
                if self.gt_ids is not None:
                    color = GREEN if ret_idx in self.gt_ids[query_idx] else RED
                else:
                    color = BLACK

                plt.subplot(n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + j + 2)
                img = visualize_gallery_fn(ret_idx, color)

                plt.title(f"Gallery #{ret_idx} - {round(self.distances[query_idx][j].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            if self.gt_ids is not None:

                for k, gt_idx in enumerate(self.gt_ids[query_idx][:n_gt_to_show]):
                    plt.subplot(
                        n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + k + n_galleries_to_show + 2
                    )

                    img = visualize_gallery_fn(gt_idx, GRAY)

                    plt.title("GT")
                    plt.imshow(img)
                    plt.axis("off")

        fig.tight_layout()

        if show:
            fig.show()

        return fig


__all__ = ["RetrievalResults"]
