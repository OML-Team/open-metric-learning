import math
from functools import partial
from random import randint
from typing import List, Optional, Tuple

import pytest
import torch
from torch import FloatTensor, LongTensor

from oml.functional.knn import batched_knn
from oml.utils.misc import one_hot
from oml.utils.misc_torch import pairwise_dist
from tests.utils import check_if_sequence_of_tensors_are_equal


def straight_knn(
    embeddings: FloatTensor,
    ids_query: LongTensor,
    ids_gallery: LongTensor,
    labels_gt: Optional[LongTensor],
    sequence_ids: Optional[LongTensor],
    top_n: int,
) -> Tuple[List[FloatTensor], List[LongTensor], Optional[List[LongTensor]]]:
    top_n = min(top_n, len(ids_gallery))

    mask_to_ignore = ids_query[..., None] == ids_gallery[None, ...]
    if sequence_ids is not None:
        mask_sequence = sequence_ids[ids_query][..., None] == sequence_ids[ids_gallery][None, ...]
        mask_to_ignore = torch.logical_or(mask_to_ignore, mask_sequence)

    distances_all = pairwise_dist(x1=embeddings[ids_query], x2=embeddings[ids_gallery], p=2)
    distances_all[mask_to_ignore] = float("inf")
    distances_mat, retrieved_ids_mat = torch.topk(distances_all, k=top_n, largest=False, sorted=True)

    distances, retrieved_ids = [], []
    for d, r in zip(distances_mat, retrieved_ids_mat):
        mask_to_keep = ~d.isinf()
        distances.append(d[mask_to_keep].view(-1))
        retrieved_ids.append(r[mask_to_keep].view(-1))

    if labels_gt is not None:
        mask_gt = labels_gt[ids_query][..., None] == labels_gt[ids_gallery][None, ...]
        mask_gt[mask_to_ignore] = False
        gt_ids = [LongTensor(row.nonzero()).view(-1) for row in mask_gt]
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
            labels_gt=labels,
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

        assert check_if_sequence_of_tensors_are_equal(distances, distances_)
        assert check_if_sequence_of_tensors_are_equal(retrieved_ids, retrieved_ids_)

        if need_gt:
            assert check_if_sequence_of_tensors_are_equal(gt_ids, gt_ids_)


@pytest.mark.parametrize("knn_func", [batched_knn, straight_knn])
def test_on_exact_case(knn_func):  # type: ignore
    oh = partial(one_hot, dim=4)

    embeddings = torch.stack([oh(0), oh(0), oh(0), oh(1), oh(1), oh(1)]).float()
    labels_gt = torch.tensor([0, 0, 0, 1, 1, 1]).long()
    sequence_id = torch.tensor([0, 1, 2, 3, 3, 4]).long()
    ids_query = torch.tensor([0, 3]).long()
    ids_gallery = torch.tensor([0, 1, 2, 3, 4, 5]).long()
    top_n = 10

    distances_expected = [
        FloatTensor([0, 0, math.sqrt(2), math.sqrt(2), math.sqrt(2)]),
        FloatTensor([0, math.sqrt(2), math.sqrt(2), math.sqrt(2)]),
    ]

    # todo: handle better sorting of the same distances because it may change with seed
    retrieved_ids_expected = [LongTensor([1, 2, 3, 4, 5]), LongTensor([5, 1, 2, 0])]

    gt_ids_expected = [LongTensor([1, 2]), LongTensor([5])]

    distances, retrieved_ids, gt_ids = knn_func(
        embeddings=embeddings,
        labels_gt=labels_gt,
        ids_query=ids_query,
        ids_gallery=ids_gallery,
        top_n=top_n,
        sequence_ids=sequence_id,
    )

    assert check_if_sequence_of_tensors_are_equal(distances_expected, distances)
    assert check_if_sequence_of_tensors_are_equal(retrieved_ids_expected, retrieved_ids_expected)
    assert check_if_sequence_of_tensors_are_equal(gt_ids_expected, gt_ids_expected)
