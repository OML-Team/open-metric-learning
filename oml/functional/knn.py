from typing import List, Optional, Tuple

import torch
from torch import FloatTensor, LongTensor

from oml.const import BS_KNN
from oml.utils.misc_torch import pairwise_dist


def batched_knn(
    embeddings: FloatTensor,
    ids_query: LongTensor,
    ids_gallery: LongTensor,
    top_n: int,
    sequence_ids: Optional[LongTensor] = None,
    labels_gt: Optional[LongTensor] = None,
    bs: int = BS_KNN,
) -> Tuple[FloatTensor, LongTensor, Optional[List[LongTensor]]]:
    """

    Args:
        embeddings: Matrix with the shape of ``[n, dim]``
        ids_query:  Tensor with the size of ``Q``, where ``Q <= n``. Each element is withing the range ``(0, n - 1)``.
        ids_gallery:  Tensor with the size of ``G`` where ``G <= n``. Each element is withing the range ``(0, n - 1)``.
                      May overlap with ``ids_query``.
        top_n: Number of neighbors to find and return.
        sequence_ids: Sequence identifiers with the size of ``n`` (if known).
        labels_gt: Ground truth labels of every element with the size of ``n`` (if known).
        bs: Batch size for computing distances to avoid OOM errors when processing the whole matrix at once.

    Returns:
        distances: Sorted distances from every query to the closest ``top_n`` galleries with the size of ``(Q, top_n)``.
        retrieved_ids: The corresponding ids of gallery items with the shape of ``(Q, top_n)``.
                       Each element is withing the range ``(0, G - 1)``.
        gt_ids: Ids of the gallery items relevant to every query. Each element is withing the range ``(0, G - 1)``.
                It's only computed if ``labels_gt`` is provided.

    """
    assert (ids_query.ndim == 1) and (ids_gallery.ndim == 1) and (embeddings.ndim == 2)
    assert len(embeddings) <= len(ids_query) + len(ids_gallery)
    assert (sequence_ids is None) or (len(sequence_ids) == len(embeddings) and (sequence_ids.ndim == 1))
    assert (labels_gt is None) or (len(labels_gt) <= len(ids_query) + len(ids_gallery) and (labels_gt.ndim == 1))

    top_n = min(top_n, len(ids_gallery))

    emb_q = embeddings[ids_query]
    emb_g = embeddings[ids_gallery]

    nq = len(ids_query)
    retrieved_ids = LongTensor(nq, top_n)
    distances = FloatTensor(nq, top_n)
    gt_ids = []

    # we do batching over first (queries) dimension
    for i in range(0, nq, bs):
        distances_b = pairwise_dist(x1=emb_q[i : i + bs], x2=emb_g)
        ids_query_b = ids_query[i : i + bs]

        mask_to_ignore_b = ids_query_b[..., None] == ids_gallery[None, ...]
        if sequence_ids is not None:
            mask_sequence = sequence_ids[ids_query_b][..., None] == sequence_ids[ids_gallery][None, ...]
            mask_to_ignore_b = torch.logical_or(mask_to_ignore_b, mask_sequence)

        if labels_gt is not None:
            mask_gt_b = labels_gt[ids_query_b][..., None] == labels_gt[ids_gallery][None, ...]
            mask_gt_b[mask_to_ignore_b] = False
            gt_ids.extend([LongTensor(row.nonzero()).view(-1) for row in mask_gt_b])  # type: ignore

        distances_b[mask_to_ignore_b] = float("inf")
        distances[i : i + bs], retrieved_ids[i : i + bs] = torch.topk(distances_b, k=top_n, largest=False, sorted=True)

    return distances, retrieved_ids, gt_ids or None


__all__ = ["batched_knn"]
