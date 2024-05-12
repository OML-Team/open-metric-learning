from typing import Optional, Tuple

import torch
from torch import FloatTensor, LongTensor, Tensor, bincount, full
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
) -> Tuple[FloatTensor, Tensor, Optional[Tensor]]:
    """
    # todo 522: CHANGE DOCS

    Args:
        embeddings: Matrix with the shape of ``[L, dim]``
        ids_query:  Tensor with the size of ``Q``, where ``Q <= n``. Each element is within the range ``(0, L - 1)``.
        ids_gallery:  Tensor with the size of ``G`` where ``G <= n``. Each element is within the range ``(0, L - 1)``.
                      May overlap with ``ids_query``.
        top_n: Number of neighbors to find and return.
        sequence_ids: Sequence identifiers with the size of ``L`` (if known).
        labels_gt: Ground truth labels of every element with the size of ``L`` (if known).
        bs: Batch size for computing distances to avoid OOM errors when processing the whole matrix at once.

    Returns:
        distances: Sorted distances from every query to the closest ``top_n`` galleries with the size of ``(Q, top_n)``.
        retrieved_ids: The corresponding ids of gallery items with the shape of ``(Q, top_n)``.
                       Each element is within the range ``(0, G - 1)``.
        gt_ids: Ids of the gallery items relevant to every query. Each element is within the range ``(0, G - 1)``.
                It's only computed if ``labels_gt`` is provided.

    """
    assert (ids_query.ndim == 1) and (ids_gallery.ndim == 1) and (embeddings.ndim == 2)
    assert len(embeddings) <= len(ids_query) + len(ids_gallery)
    assert (sequence_ids is None) or ((len(sequence_ids) == len(embeddings)) and (sequence_ids.ndim == 1))
    assert (labels_gt is None) or ((len(labels_gt) == embeddings.shape[0]) and (labels_gt.ndim == 1))

    top_n = min(top_n, len(ids_gallery))

    emb_q = embeddings[ids_query]
    emb_g = embeddings[ids_gallery]

    nq = len(ids_query)
    retrieved_ids = Tensor(nq, top_n)
    distances = FloatTensor(nq, top_n)

    gt_ids = None if labels_gt is None else full((nq, bincount(labels_gt).max() - 1), float("nan"))

    # since we don't now the size of mask_to_ignore in advance, we probably pre allocated some extra
    # memory for gt ids, that is why we need to keep track of max seen number of gts for further clipping
    max_seen_n_gt = 0

    # we do batching over first (queries) dimension
    for i in tqdm(range(0, nq, bs), desc="Batched KNN."):
        distances_b = pairwise_dist(x1=emb_q[i : i + bs, :], x2=emb_g)
        ids_query_b = ids_query[i : i + bs]

        # the logic behind: we want to ignore the item during search if it was used for both: query and gallery
        mask_to_ignore_b = ids_query_b[..., None] == ids_gallery[None, ...]
        if sequence_ids is not None:
            # sometimes our items may be packed into the groups, so we ignore other members of this group during search
            # more info in the docs: find for "Handling sequences of photos"
            mask_sequence = sequence_ids[ids_query_b][..., None] == sequence_ids[ids_gallery][None, ...]
            mask_to_ignore_b = torch.logical_or(mask_to_ignore_b, mask_sequence)

        if labels_gt is not None:
            mask_gt_b = labels_gt[ids_query_b][..., None] == labels_gt[ids_gallery][None, ...]
            mask_gt_b[mask_to_ignore_b] = False
            for k in range(i, min(nq, i + bs)):
                gt_ids_query = mask_gt_b[k - i, :].nonzero()
                gt_ids[k, : len(gt_ids_query)] = gt_ids_query.view(-1)
                max_seen_n_gt = max(max_seen_n_gt, len(gt_ids_query))

        distances_b[mask_to_ignore_b] = float("nan")
        dists, ids = torch.topk(distances_b, k=top_n, largest=False, sorted=True)
        ids = ids.float()
        ids[dists.isnan()] = float("nan")

        distances[i : i + bs, :], retrieved_ids[i : i + bs, :] = dists, ids

    if labels_gt is not None:
        gt_ids = gt_ids[:, :max_seen_n_gt]

    return distances, retrieved_ids, gt_ids


__all__ = ["batched_knn"]
