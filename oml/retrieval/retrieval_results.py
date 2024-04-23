from typing import List

import matplotlib.pyplot as plt
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
        distances: FloatTensor,
        retrieved_ids: LongTensor,
        gt_ids: List[LongTensor] = None,
    ):
        """
        Args:
            distances: Sorted distances to the first ``top_n`` gallery items with the shape of ``[n_query, top_n]``.
            retrieved_ids: Top N gallery ids retrieved for every query with the shape of ``[n_query, top_n]``.
                Every element is within the range ``(0, n_gallery - 1)``.
            gt_ids: Gallery ids relevant to every query, list of ``n_query`` elements where every element may
                have an arbitrary length. Every element is within the range ``(0, n_gallery - 1)``
        """
        assert distances.shape == retrieved_ids.shape

        if gt_ids is not None:
            assert distances.shape[0] == len(gt_ids)
            if not all(len(x) > 0 for x in gt_ids):
                raise RuntimeError("Every query must have at least one relevant gallery id.")

        self.distances = distances
        self.retrieved_ids = retrieved_ids
        self.gt_ids = gt_ids

    @property
    def top_n(self) -> int:
        return self.retrieved_ids.shape[1]

    @classmethod
    def compute_from_embeddings(
        cls,
        embeddings: FloatTensor,
        dataset: IQueryGalleryDataset,
        n_items_to_retrieve: int = 1_000,
    ) -> "RetrievalResults":

        # todo 522: rework
        ignoring_groups = dataset.df[SEQUENCE_COLUMN] if SEQUENCE_COLUMN in dataset.df else None

        if isinstance(dataset, IQueryGalleryLabeledDataset):
            labels_gt = dataset.get_labels()
        else:
            labels_gt = None

        distances, retrieved_ids, gt_ids = batched_knn(
            embeddings=embeddings,
            ids_query=dataset.get_query_ids(),
            ids_gallery=dataset.get_gallery_ids(),
            labels_gt=labels_gt,
            sequence_ids=ignoring_groups,
            top_n=n_items_to_retrieve,
        )

        return RetrievalResults(distances=distances, retrieved_ids=retrieved_ids, gt_ids=gt_ids)

    def __repr__(self) -> str:
        txt = (
            f"You retrieved {self.top_n} items.\n"
            f"Distances to the retrieved items:\n{self.distances}.\n"
            f"Ids of the retrieved gallery items:\n{self.retrieved_ids}.\n"
        )

        if self.gt_ids is None:
            txt += "Ground truths are unknown.\n"
        else:
            gt_ids_list = [x.tolist() for x in self.gt_ids]
            txt += f"Ground truth gallery ids are:\n{gt_ids_list}.\n"

        return txt

    def visualize(
        self,
        query_ids: List[int],
        dataset: IQueryGalleryDataset,
        n_galleries_to_show: int = 5,
        verbose: bool = False,
    ) -> plt.Figure:
        if not isinstance(dataset, (IVisualizableDataset, IQueryGalleryDataset)):
            raise ValueError(
                f"Dataset has to support {IVisualizableDataset.__name__} and "
                f"{IQueryGalleryDataset} interfaces. Got {type(dataset)}."
            )

        if verbose:
            # todo: add something smarter later
            print(f"Visualizing {n_galleries_to_show} for the following query ids: {query_ids}.")

        ii_query = dataset.get_query_ids()
        ii_gallery = dataset.get_gallery_ids()

        n_galleries_to_show = min(n_galleries_to_show, self.distances.shape[1])

        n_gt = N_GT_SHOW_EMBEDDING_METRICS if (self.gt_ids is not None) else 0

        fig = plt.figure(figsize=(16, 16 / (n_galleries_to_show + n_gt + 1) * len(query_ids)))

        n_rows, n_cols = len(query_ids), n_galleries_to_show + 1 + N_GT_SHOW_EMBEDDING_METRICS

        # iterate over queries
        for j, query_idx in enumerate(query_ids):

            plt.subplot(n_rows, n_cols, j * (n_galleries_to_show + 1 + n_gt) + 1)

            img = dataset.visualize(item=ii_query[query_idx].item(), color=BLUE)

            plt.imshow(img)
            plt.title("Query")
            plt.axis("off")

            # iterate over retrieved items
            for i, ret_idx in enumerate(self.retrieved_ids[query_idx][:n_galleries_to_show]):
                if self.gt_ids is not None:
                    color = GREEN if ret_idx in self.gt_ids[query_idx] else RED
                else:
                    color = BLACK

                plt.subplot(n_rows, n_cols, j * (n_galleries_to_show + 1 + n_gt) + i + 2)
                img = dataset.visualize(item=ii_gallery[ret_idx].item(), color=color)

                print(self.distances.shape, "vvvv")
                plt.title(f"{i} - {round(self.distances[query_idx, ret_idx].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            if self.gt_ids is not None:

                for k, gt_idx in enumerate(self.gt_ids[query_idx][:N_GT_SHOW_EMBEDDING_METRICS]):
                    plt.subplot(n_rows, n_cols, j * (n_galleries_to_show + 1 + n_gt) + k + n_galleries_to_show + 2)

                    img = dataset.visualize(item=ii_gallery[gt_idx].item(), color=GRAY)
                    plt.title("GT " + str(round(self.distances[query_idx, gt_idx].item(), 3)))
                    plt.imshow(img)
                    plt.axis("off")

        fig.tight_layout()
        return fig


__all__ = ["RetrievalResults"]
