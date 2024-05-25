from typing import Optional, Sequence, Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor
from tqdm.auto import tqdm

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
    verbose: bool = False,
) -> Tuple[Sequence[FloatTensor], Sequence[LongTensor], Optional[Sequence[LongTensor]]]:
    """

    Args:
        embeddings: Matrix with the shape of ``[L, dim]``
        ids_query:  Tensor with the size of ``Q``, where ``Q <= n``. Each element is within the range ``(0, L - 1)``.
        ids_gallery:  Tensor with the size of ``G`` where ``G <= n``. Each element is within the range ``(0, L - 1)``.
                      May overlap with ``ids_query``.
        top_n: Number of neighbors to retrieve.
        sequence_ids: Sequence identifiers with the size of ``L`` (if known).
        labels_gt: Ground truth labels of every element with the size of ``L`` (if known).
        bs: Batch size for computing distances to avoid OOM errors when processing the whole matrix at once.
        verbose: Set ``True`` to see progress bar.

    Returns:
        distances: Sorted distances from queries to the first gallery items with the size of ``Q``.
        retrieved_ids: First gallery indices retrieved for every query with the size of ``Q``.
                Every index is within the range ``(0, G - 1)``.
        gt_ids: Gallery indices relevant to every query with the size of ``Q``.
                Every element is within the range ``(0, G - 1)``

    """
    assert (ids_query.ndim == 1) and (ids_gallery.ndim == 1) and (embeddings.ndim == 2)
    assert len(embeddings) <= len(ids_query) + len(ids_gallery)
    assert (sequence_ids is None) or ((len(sequence_ids) == len(embeddings)) and (sequence_ids.ndim == 1))
    assert (labels_gt is None) or ((len(labels_gt) == embeddings.shape[0]) and (labels_gt.ndim == 1))

    top_n = min(top_n, len(ids_gallery))

    embeddings_query = embeddings[ids_query]
    embeddings_gallery = embeddings[ids_gallery]

    nq = len(ids_query)

    retrieved_ids = []
    distances = []
    gt_ids = []

    # we do batching over first (queries) dimension
    items = tqdm(range(0, nq, bs), desc="Finding nearest neighbors.") if verbose else range(0, nq, bs)
    for i in items:
        distances_b = pairwise_dist(x1=embeddings_query[i : i + bs, :], x2=embeddings_gallery, p=2)
        ids_query_b = ids_query[i : i + bs]

        # we want to ignore the item during search if it was used for both: query and gallery
        mask_to_ignore_b = ids_query_b[..., None] == ids_gallery[None, ...]
        if sequence_ids is not None:
            # our items may be packed into the sequences, so we ignore other members of this sequence during search
            # more info in the docs: search for "Handling sequences of photos"
            mask_sequence = sequence_ids[ids_query_b][..., None] == sequence_ids[ids_gallery][None, ...]
            mask_to_ignore_b = torch.logical_or(mask_to_ignore_b, mask_sequence)

        if labels_gt is not None:
            mask_gt_b = BoolTensor(labels_gt[ids_query_b][..., None] == labels_gt[ids_gallery][None, ...])
            mask_gt_b[mask_to_ignore_b] = False
            gt_ids.extend([row.nonzero().view(-1) for row in mask_gt_b])  # type: ignore

        distances_b[mask_to_ignore_b] = float("inf")
        distances_b_sorted, retrieved_ids_b = torch.topk(distances_b, k=top_n, largest=False, sorted=True)

        # every query may have arbitrary number of retrieved items, so we are forced to use a loop to store the results
        for dist, ids in zip(distances_b_sorted, retrieved_ids_b):
            mask_to_keep = ~dist.isinf()
            distances.append(dist[mask_to_keep].view(-1))
            retrieved_ids.append(ids[mask_to_keep].view(-1))

    return distances, retrieved_ids, gt_ids or None


__all__ = ["batched_knn"]
