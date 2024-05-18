from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from torch import FloatTensor, LongTensor

from oml.const import (
    BLACK,
    BLUE,
    GRAY,
    GREEN,
    N_GT_SHOW_EMBEDDING_METRICS,
    RED,
    SEQUENCE_COLUMN,
)
from oml.functional.knn import batched_knn
from oml.interfaces.datasets import (
    IQueryGalleryDataset,
    IQueryGalleryLabeledDataset,
    IVisualizableDataset,
)


class RetrievalResults:
    def __init__(
        self,
        distances: List[FloatTensor],
        retrieved_ids: List[LongTensor],
        gt_ids: List[LongTensor] = None,
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
            assert (d[:-1] <= d[1:]).all(), "Distances must be sorted."
            assert len(set(r)) == len(r), "Retrieved ids must be unique."
            assert len(d) == len(r) > 0

        if gt_ids is not None:
            assert len(distances) == len(gt_ids)
            if any(len(x) == 0 for x in gt_ids):
                raise RuntimeError("Every query must have at least one relevant gallery id.")

        self.distances = distances
        self.retrieved_ids = retrieved_ids
        self.gt_ids = gt_ids

    @property
    def n_retrieved_items(self) -> int:
        """
        Returns: Number of items retrieved for each query. If queries have different number of retrieved items,
            returns the maximum of them.

        """
        return max(len(x) for x in self.retrieved_ids)

    @classmethod
    def compute_from_embeddings(
        cls,
        embeddings: FloatTensor,
        dataset: IQueryGalleryDataset,
        n_items_to_retrieve: int = 100,
    ) -> "RetrievalResults":
        """
        Args:
            embeddings: The result of inference with the shape of ``[dataset_len, emb_dim]``.
            dataset: Dataset having query/gallery split.
            n_items_to_retrieve: Number of the closest gallery items to retrieve. It may be clipped by
                gallery size if needed. Note, some queries may get less than this number of retrieved items if they
                don't have enough gallery items available.

        """
        assert len(embeddings) == len(dataset), "Embeddings and dataset must have the same size."

        if SEQUENCE_COLUMN in dataset.extra_data:
            sequence_ids = LongTensor(pd.factorize(dataset.extra_data[SEQUENCE_COLUMN])[0])
        else:
            sequence_ids = None

        labels_gt = dataset.get_labels() if isinstance(dataset, IQueryGalleryLabeledDataset) else None

        distances, retrieved_ids, gt_ids = batched_knn(
            embeddings=embeddings,
            ids_query=dataset.get_query_ids(),
            ids_gallery=dataset.get_gallery_ids(),
            labels_gt=labels_gt,
            sequence_ids=sequence_ids,
            top_n=n_items_to_retrieve,
        )

        return RetrievalResults(distances=distances, retrieved_ids=retrieved_ids, gt_ids=gt_ids)

    def __str__(self) -> str:
        max_el_to_show = 100

        txt = (
            f"You retrieved {self.n_retrieved_items} items.\n"
            f"Distances to the retrieved items:\n{self.distances[:max_el_to_show]}.\n"
            f"Ids of the retrieved gallery items:\n{self.retrieved_ids[:max_el_to_show]}.\n"
        )

        if self.gt_ids is None:
            txt += "Ground truths are unknown.\n"
        else:
            gt_ids_list = [x.tolist() for x in self.gt_ids]
            txt += f"Ground truth gallery ids are:\n{gt_ids_list[:max_el_to_show]}.\n"

        return txt

    def visualize(
        self,
        query_ids: List[int],
        dataset: IQueryGalleryDataset,
        n_galleries_to_show: int = 5,
        n_gt_to_show: int = N_GT_SHOW_EMBEDDING_METRICS,
        verbose: bool = False,
    ) -> plt.Figure:
        """
        Args:
            query_ids: Query indices within the range of ``(0, n_query - 1)``.
            dataset: Dataset that provides query-gallery split and supports visualisation.
            n_galleries_to_show: Number of closest gallery items to show.
            n_gt_to_show: Number of ground truth gallery items to show for reference (if available).
            verbose: Set ``True`` to allow prints.

        """
        if not isinstance(dataset, IVisualizableDataset):
            raise TypeError(f"Dataset has to support {IVisualizableDataset.__name__}. Got {type(dataset)}.")
        if not isinstance(dataset, IQueryGalleryDataset):
            raise TypeError(f"Dataset has to support {IQueryGalleryDataset.__name__}. Got {type(dataset)}.")

        if verbose:
            print(f"Visualizing {n_galleries_to_show} for the following query ids: {query_ids}.")

        ii_query = dataset.get_query_ids()
        ii_gallery = dataset.get_gallery_ids()

        n_galleries_to_show = min(n_galleries_to_show, self.n_retrieved_items)
        n_gt_to_show = n_gt_to_show if (self.gt_ids is not None) else 0

        fig = plt.figure(figsize=(16, 16 / (n_galleries_to_show + n_gt_to_show + 1) * len(query_ids)))
        n_rows, n_cols = len(query_ids), n_galleries_to_show + 1 + n_gt_to_show

        # iterate over queries
        for i, query_idx in enumerate(query_ids):

            plt.subplot(n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + 1)

            img = dataset.visualize(item=ii_query[query_idx].item(), color=BLUE)

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
                img = dataset.visualize(item=ii_gallery[ret_idx].item(), color=color)

                plt.title(f"Gallery #{ret_idx} - {round(self.distances[query_idx][j].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            if self.gt_ids is not None:

                for k, gt_idx in enumerate(self.gt_ids[query_idx][:n_gt_to_show]):
                    plt.subplot(
                        n_rows, n_cols, i * (n_galleries_to_show + 1 + n_gt_to_show) + k + n_galleries_to_show + 2
                    )

                    img = dataset.visualize(item=ii_gallery[gt_idx].item(), color=GRAY)
                    plt.title("GT")
                    plt.imshow(img)
                    plt.axis("off")

        fig.tight_layout()
        return fig


__all__ = ["RetrievalResults"]
