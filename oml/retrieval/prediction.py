from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import FloatTensor, LongTensor

from oml.const import (
    BLACK,
    BLUE,
    BS_KNN,
    GRAY,
    GREEN,
    N_GT_SHOW_EMBEDDING_METRICS,
    RED,
    SEQUENCE_COLUMN,
)
from oml.interfaces.datasets import IDatasetQueryGallery, IVisualizableDataset
from oml.utils.misc_torch import pairwise_dist


def batched_knn(
    embeddings: FloatTensor,
    ids_query: LongTensor,
    ids_gallery: LongTensor,
    top_n: int,
    ignoring_groups: Optional[np.ndarray] = None,
    labels_gt: Optional[np.ndarray] = None,
    bs: int = BS_KNN,
) -> Tuple[FloatTensor, LongTensor, Optional[List[LongTensor]]]:
    """

    Args:
        embeddings: Matrix with the shape of ``[n, dim]``
        ids_query:  Tensor with the size of ``Q``, where ``Q <= n``. Each element is withing the range ``(0, n - 1)``.
        ids_gallery:  Tensor with the size of ``G`` where ``G <= n``. Each element is withing the range ``(0, n - 1)``.
                      May overlap with ``ids_query``.
        top_n: Number of neighbors to keep.
        ignoring_groups: Tensor with the size of ``n`` if known.
        labels_gt: Tensor with the size of ``n`` if known.
        bs: Batch size for computing distances to avoid OOM errors.

    Returns:
        distances: Distances to the ``top_n`` neighbors with the size of ``(Q, top_n)``.
        retrieved_ids ids: Ids of the top ``top_n`` neighbors with the size of ``(Q, top_n)``.
        gt_ids: Ids of the gallery items relevant to every query. It's only computed if ``labels_gt``
                had been provided.

    """
    assert (ids_query.ndim == 1) and (ids_gallery.ndim == 1) and (embeddings.ndim == 2)
    assert len(embeddings) <= len(ids_query) + len(ids_gallery)
    assert (ignoring_groups is None) or (len(ignoring_groups) == len(embeddings))
    assert (labels_gt is None) or (len(labels_gt) <= len(ids_query) + len(ids_gallery))

    top_n = min(top_n, len(ids_gallery))

    emb_q = embeddings[ids_query]
    emb_g = embeddings[ids_gallery]

    nq = len(emb_q)
    retrieved_ids = LongTensor(nq, top_n)
    distances = FloatTensor(nq, top_n)
    gt_ids = []

    for i in range(0, nq, bs):
        distances_b = pairwise_dist(x1=emb_q[i : i + bs], x2=emb_g)
        ids_query_b = ids_query[i : i + bs]

        mask_to_ignore_b = ids_query_b[..., None] == ids_gallery[None, ...]
        if ignoring_groups is not None:
            mask_groups = ignoring_groups[ids_query_b][..., None] == ignoring_groups[ids_gallery][None, ...]
            mask_to_ignore_b = np.logical_or(mask_to_ignore_b, mask_groups).bool()

        if labels_gt is not None:
            mask_gt_b = labels_gt[ids_query_b][..., None] == labels_gt[ids_gallery][None, ...]
            mask_gt_b[mask_to_ignore_b] = False
            gt_ids.extend([LongTensor(row.nonzero()[0]) for row in mask_gt_b])  # type: ignore

        distances_b[mask_to_ignore_b] = float("inf")
        distances[i : i + bs], retrieved_ids[i : i + bs] = torch.topk(distances_b, k=top_n, largest=False, sorted=True)

    return distances, retrieved_ids, gt_ids or None


class RetrievalPrediction:
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
        dataset: IDatasetQueryGallery,
        n_items_to_retrieve: int = 1_000,
    ) -> "RetrievalPrediction":
        ignoring_groups = dataset.extra_data.get(SEQUENCE_COLUMN, None)

        distances, retrieved_ids, gt_ids = batched_knn(
            embeddings=embeddings,
            ids_query=dataset.get_query_ids(),
            ids_gallery=dataset.get_gallery_ids(),
            labels_gt=dataset.get_labels(),
            ignoring_groups=ignoring_groups,
            top_n=n_items_to_retrieve,
        )

        return RetrievalPrediction(distances=distances, retrieved_ids=retrieved_ids, gt_ids=gt_ids)

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
        dataset: IDatasetQueryGallery,
        n_galleries_to_show: int = 5,
        verbose: bool = False,
    ) -> plt.Figure:
        if not isinstance(dataset, IVisualizableDataset):
            raise ValueError(f"Dataset has to support {IVisualizableDataset.__name__}. Got {type(dataset)}.")

        if not isinstance(dataset, IDatasetQueryGallery):
            raise ValueError(f"Dataset has to support {IDatasetQueryGallery.__name__}. Got {type(dataset)}.")

        if verbose:
            # todo: add something smarter later
            print(f"Visualizing {n_galleries_to_show} for the following query ids: {query_ids}.")

        ii_query = dataset.get_query_ids()
        ii_gallery = dataset.get_gallery_ids()

        n_gt = N_GT_SHOW_EMBEDDING_METRICS if (self.gt_ids is not None) else 0

        fig = plt.figure(figsize=(16, 16 / (n_galleries_to_show + n_gt + 1) * len(query_ids)))

        n_rows, n_cols = len(query_ids), n_galleries_to_show + 1 + N_GT_SHOW_EMBEDDING_METRICS
        print("zzz", n_rows, n_cols)

        # iterate over queries
        for j, query_idx in enumerate(query_ids):

            plt.subplot(n_rows, n_cols, j * (n_galleries_to_show + 1 + n_gt) + 1)

            img = dataset.visualize(ii_query[query_idx], color=BLUE)

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
                img = dataset.visualize(ii_gallery[ret_idx], color=color)

                plt.title(f"{i} - {round(self.distances[query_idx, ret_idx].item(), 3)}")
                plt.imshow(img)
                plt.axis("off")

            if self.gt_ids is not None:

                for k, gt_idx in enumerate(self.gt_ids[query_idx][:N_GT_SHOW_EMBEDDING_METRICS]):
                    plt.subplot(n_rows, n_cols, j * (n_galleries_to_show + 1 + n_gt) + k + n_galleries_to_show + 2)

                    img = dataset.visualize(ii_gallery[gt_idx], color=GRAY)
                    plt.title("GT " + str(round(self.distances[query_idx, gt_idx].item(), 3)))
                    plt.imshow(img)
                    plt.axis("off")

        fig.tight_layout()
        return fig


__all__ = ["batched_knn", "RetrievalPrediction"]
