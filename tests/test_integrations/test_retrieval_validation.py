from math import isclose
from typing import Tuple

import pytest
import torch
from torch import BoolTensor, FloatTensor, LongTensor
from torch.utils.data import DataLoader

from oml.const import EMBEDDINGS_KEY, INPUT_TENSORS_KEY, OVERALL_CATEGORIES_KEY
from oml.metrics.embeddings import EmbeddingMetrics
from tests.test_integrations.utils import (
    EmbeddingsQueryGalleryLabeledDataset,
    IdealClusterEncoder,
)

TData = Tuple[LongTensor, BoolTensor, BoolTensor, FloatTensor, float]


def get_separate_query_gallery() -> TData:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 1])
    gallery_mask = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1, 0])
    query_mask = torch.logical_not(gallery_mask)

    # let's add some errors (swap for some labels) in queries
    input_tensors = labels.clone()
    input_tensors[4] = 0
    input_tensors[5] = 0

    cmc_gt = 3 / 5

    return labels.long(), query_mask.bool(), gallery_mask.bool(), input_tensors, cmc_gt


def get_shared_query_gallery() -> TData:
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]).long()
    query_mask = torch.ones_like(labels)
    gallery_mask = torch.ones_like(labels)

    input_tensors = labels.clone()
    input_tensors[0] = 3

    cmc_gt = 7 / 8

    return labels.long(), query_mask.bool(), gallery_mask.bool(), input_tensors, cmc_gt


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("data", [get_separate_query_gallery(), get_shared_query_gallery()])
def test_retrieval_validation(batch_size: int, shuffle: bool, num_workers: int, data: TData) -> None:
    labels, query_mask, gallery_mask, input_tensors, cmc_gt = data

    dataset = EmbeddingsQueryGalleryLabeledDataset(
        labels=labels,
        embeddings=input_tensors,
        is_query=query_mask,
        is_gallery=gallery_mask,
    )

    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers
    )

    calc = EmbeddingMetrics(cmc_top_k=(1,))
    calc.setup(num_samples=len(dataset))

    model = IdealClusterEncoder()

    for batch in loader:
        output = model(batch[INPUT_TENSORS_KEY])
        batch[EMBEDDINGS_KEY] = output
        calc.update_data(data_dict=batch)

    metrics = calc.compute_metrics()

    assert isclose(metrics[OVERALL_CATEGORIES_KEY]["cmc"][1], cmc_gt, abs_tol=1e-5)
