from random import randint
from typing import Optional, Tuple

import pytest
import torch
from torch import FloatTensor, LongTensor

from oml.functional.knn import batched_knn
from oml.utils.misc_torch import pairwise_dist


def straight_knn(
    embeddings: FloatTensor,
    ids_query: LongTensor,
    ids_gallery: LongTensor,
    labels: Optional[LongTensor],
    sequence_ids: Optional[LongTensor],
    top_n: int,
) -> Tuple[FloatTensor, LongTensor, Optional[LongTensor]]:
    top_n = min(top_n, len(ids_gallery))

    mask_to_ignore = ids_query[..., None] == ids_gallery[None, ...]
    if sequence_ids is not None:
        mask_sequence = sequence_ids[ids_query][..., None] == sequence_ids[ids_gallery][None, ...]
        mask_to_ignore = torch.logical_or(mask_to_ignore, mask_sequence)

    distances_all = pairwise_dist(x1=embeddings[ids_query], x2=embeddings[ids_gallery], p=2)
    distances_all[mask_to_ignore] = float("inf")
    distances, retrieved_ids = torch.topk(distances_all, k=top_n, largest=False, sorted=True)

    if labels is not None:
        mask_gt = labels[ids_query][..., None] == labels[ids_gallery][None, ...]
        mask_gt[mask_to_ignore] = False
        gt_ids = [LongTensor(row.nonzero()).squeeze() for row in mask_gt]
    else:
        gt_ids = None

    return distances, retrieved_ids, gt_ids


def generate_data(
    dataset_len: int, n_classes: Optional[int], n_sequences: Optional[int], separate_query_gallery: bool
) -> Tuple[FloatTensor, LongTensor, LongTensor, Optional[LongTensor], Optional[LongTensor]]:
    if separate_query_gallery:
        n_query = randint(1, dataset_len - 1)
        ii = torch.randperm(n=dataset_len)
        ids_query, ids_gallery = ii[:n_query], ii[n_query:]

    else:
        n_query = randint(1 + dataset_len // 2, dataset_len)
        n_gallery = randint(1 + dataset_len // 2, dataset_len)
        ids_query = torch.randperm(n=dataset_len)[:n_query]
        ids_gallery = torch.randperm(n=dataset_len)[:n_gallery]
        assert set(ids_query.tolist()).intersection(ids_gallery.tolist()), "Query and gallery don't intersect!"

    embeddings = torch.randn((dataset_len, 8)).float()
    labels = torch.randint(0, n_classes, size=(dataset_len,)) if n_classes else None
    sequence_ids = torch.randint(0, n_sequences, size=(dataset_len,)) if n_sequences else None

    return embeddings, ids_query, ids_gallery, labels, sequence_ids


@pytest.mark.parametrize("dataset_len", [2, 10, 30])
@pytest.mark.parametrize("need_sequence", [True, False])
@pytest.mark.parametrize("need_gt", [True, False])
@pytest.mark.parametrize("separate_query_gallery", [True, False])
def test_batched_knn(dataset_len: int, need_sequence: bool, need_gt: bool, separate_query_gallery: bool) -> None:
    for i in range(5):
        batch_size_knn = randint(1, dataset_len)
        n_classes = randint(1, 5) if need_gt else None
        n_sequences = randint(1, 4) if need_sequence else None
        top_n = randint(1, int(1.5 * dataset_len))

        embeddings, ids_query, ids_gallery, labels, sequence_ids = generate_data(
            dataset_len=dataset_len,
            n_classes=n_classes,
            n_sequences=n_sequences,
            separate_query_gallery=separate_query_gallery,
        )

        distances_, retrieved_ids_, gt_ids_ = straight_knn(
            embeddings=embeddings,
            ids_query=ids_query,
            ids_gallery=ids_gallery,
            top_n=top_n,
            labels=labels,
            sequence_ids=sequence_ids,
        )

        distances, retrieved_ids, gt_ids = batched_knn(
            embeddings=embeddings,
            ids_query=ids_query,
            ids_gallery=ids_gallery,
            top_n=top_n,
            labels_gt=labels,
            bs=batch_size_knn,
            sequence_ids=sequence_ids,
        )

        assert torch.allclose(distances, distances_)
        assert (retrieved_ids == retrieved_ids_).all()

        if need_gt:
            for (ii, ii_) in zip(gt_ids, gt_ids_):
                assert (ii == ii_).all()
