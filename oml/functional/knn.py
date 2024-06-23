from typing import Optional, Sequence, Tuple

import torch
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from tqdm.auto import tqdm

from oml.const import BS_KNN
from oml.utils.misc_torch import pairwise_dist


def check_both_are_nones_or_not_nones(a: Optional[Tensor], b: Optional[Tensor]) -> bool:
    return (a is not None and b is not None) or (a is None and b is None)


def check_len_and_ndim(a: Optional[Tensor], expected_len: int, expected_ndim: int) -> bool:
    return (a is None) or (len(a) == expected_len and a.ndim == expected_ndim)


def batched_knn_qg(
    embeddings_query: FloatTensor,
    embeddings_gallery: FloatTensor,
    top_n: int,
    labels_query: Optional[LongTensor] = None,
    labels_gallery: Optional[LongTensor] = None,
    ids_query: Optional[LongTensor] = None,
    ids_gallery: Optional[LongTensor] = None,
    sequence_ids_query: Optional[LongTensor] = None,
    sequence_ids_gallery: Optional[LongTensor] = None,
    bs: int = BS_KNN,
    verbose: bool = False,
) -> Tuple[Sequence[FloatTensor], Sequence[LongTensor], Optional[Sequence[LongTensor]]]:
    """
    Arguments have the same meaning as in `batched_knn`.

    Note, since queries and galleries are separated we additionally need
    `ids_query` and `ids_gallery` in order to process the situation when the same items
    are in the query and are in the gallery. If `ids_query` and `ids_gallery` don't overlap,
    just set both of them to Nones.

    """
    nq = len(embeddings_query)
    ng = len(embeddings_gallery)

    assert (embeddings_query.ndim == 2) and (embeddings_gallery.ndim == 2)
    assert check_both_are_nones_or_not_nones(labels_query, labels_gallery)
    assert check_both_are_nones_or_not_nones(ids_query, ids_gallery)
    assert check_both_are_nones_or_not_nones(sequence_ids_query, sequence_ids_gallery)

    assert check_len_and_ndim(labels_query, nq, 1)
    assert check_len_and_ndim(labels_gallery, ng, 1)
    assert check_len_and_ndim(ids_query, nq, 1)
    assert check_len_and_ndim(ids_gallery, ng, 1)
    assert check_len_and_ndim(sequence_ids_query, nq, 1)
    assert check_len_and_ndim(sequence_ids_gallery, ng, 1)

    top_n = min(top_n, ng)

    retrieved_ids = []
    distances = []
    gt_ids = []

    # we do batching over first (queries) dimension
    items = tqdm(range(0, nq, bs), desc="Finding nearest neighbors.") if verbose else range(0, nq, bs)
    for i in items:
        distances_b = pairwise_dist(x1=embeddings_query[i : i + bs, :], x2=embeddings_gallery, p=2)

        mask_to_ignore_b = torch.zeros_like(distances_b, dtype=torch.bool).to(distances_b.device)

        if (ids_query is not None) and (ids_gallery is not None):
            # we want to ignore the item during search if it was used for both: query and gallery
            mask_to_ignore_same_item = ids_query[i : i + bs][..., None] == ids_gallery[None, ...]
            mask_to_ignore_b = torch.logical_or(mask_to_ignore_b, mask_to_ignore_same_item)

        if (sequence_ids_query is not None) and (sequence_ids_gallery is not None):
            # our items may be packed into the sequences, so we ignore other members of this sequence during search
            # more info in the docs: search for "Handling sequences of photos"
            mask_sequence = sequence_ids_query[i : i + bs][..., None] == sequence_ids_gallery[None, ...]
            mask_to_ignore_b = torch.logical_or(mask_to_ignore_b, mask_sequence)

        if (labels_query is not None) and (labels_gallery is not None):
            mask_gt_b = BoolTensor(labels_query[i : i + bs][..., None] == labels_gallery[None, ...])
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
    assert check_len_and_ndim(sequence_ids, len(embeddings), 1)
    assert check_len_and_ndim(labels_gt, len(embeddings), 1)

    return batched_knn_qg(
        embeddings_query=embeddings[ids_query],
        embeddings_gallery=embeddings[ids_gallery],
        ids_query=ids_query,
        ids_gallery=ids_gallery,
        top_n=top_n,
        bs=bs,
        verbose=verbose,
        labels_query=None if labels_gt is None else labels_gt[ids_query],
        labels_gallery=None if labels_gt is None else labels_gt[ids_gallery],
        sequence_ids_query=None if sequence_ids is None else sequence_ids[ids_query],
        sequence_ids_gallery=None if sequence_ids is None else sequence_ids[ids_gallery],
    )


__all__ = ["batched_knn", "batched_knn_qg"]
