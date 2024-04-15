from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import BoolTensor, FloatTensor, LongTensor

from oml.const import BS_KNN, SEQUENCE_COLUMN
from oml.interfaces.datasets import IDatasetQueryGallery
from oml.utils.misc_torch import pairwise_dist


def validate_dataset(mask_gt: BoolTensor, mask_to_ignore: BoolTensor) -> None:
    is_valid = (mask_gt & ~mask_to_ignore).any(1).all()

    if not is_valid:
        raise RuntimeError("There are queries without available correct answers in the gallery!")


def batched_knn(
    embeddings: FloatTensor,
    ids_query: LongTensor,
    ids_gallery: LongTensor,
    top_n: int,
    ignoring_groups: Optional[np.ndarray] = None,
    labels_gt: Optional[LongTensor] = None,
    bs: int = BS_KNN,
) -> Tuple[FloatTensor, LongTensor, Optional[List[LongTensor]]]:
    assert (ids_query.ndim == 1) and (ids_gallery.ndim == 1) and (embeddings.ndim == 2)
    assert len(embeddings) <= len(ids_query) + len(ids_gallery)
    assert (ignoring_groups is None) or (len(ignoring_groups) == len(ids_query))
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
            gt_ids.extend([torch.nonzero(row, as_tuple=True)[0].long() for row in mask_gt_b])  # type: ignore

            validate_dataset(mask_gt_b, mask_to_ignore_b)  # type: ignore

        distances_b[mask_to_ignore_b] = float("inf")
        a, b = torch.topk(distances_b, k=top_n, largest=False, sorted=True)  # todo
        distances[i : i + bs] = a
        retrieved_ids[i : i + bs] = b

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

        self.distances = distances
        self.retrieved_ids = retrieved_ids
        self.gt_ids = gt_ids

    @classmethod
    def compute_from_embeddings(
        cls,
        embeddings: FloatTensor,
        dataset: IDatasetQueryGallery,
        n_ids_to_retrieve: int = 500,
    ) -> "RetrievalPrediction":
        # todo 522: support sequence ids properly and naming
        if hasattr(dataset, "df") and SEQUENCE_COLUMN in dataset.df:
            ignoring_groups = np.array(dataset.df[SEQUENCE_COLUMN])
        else:
            ignoring_groups = None

        distances, retrieved_ids, gt_ids = batched_knn(
            embeddings=embeddings,
            ids_query=dataset.get_query_ids(),
            ids_gallery=dataset.get_gallery_ids(),
            labels_gt=dataset.get_labels(),
            ignoring_groups=ignoring_groups,
            top_n=n_ids_to_retrieve,
        )

        return RetrievalPrediction(distances=distances, retrieved_ids=retrieved_ids, gt_ids=gt_ids)
