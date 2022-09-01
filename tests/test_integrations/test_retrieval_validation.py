from math import isclose
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from oml.const import (
    EMBEDDINGS_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    OVERALL_CATEGORIES_KEY,
)
from oml.interfaces.datasets import IDatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from tests.test_integrations.utils import IdealClusterEncoder

TData = Tuple[Tensor, Tensor, Tensor, Tensor, float]


def get_separate_query_gallery() -> TData:
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 1])
    gallery_mask = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1, 0])
    query_mask = torch.logical_not(gallery_mask)

    # let's add some errors (swap for some labels) in queries
    input_tensors = labels.clone()
    input_tensors[4] = 0
    input_tensors[5] = 0

    cmc_gt = 3 / 5

    return labels, query_mask, gallery_mask, input_tensors, cmc_gt


def get_shared_query_gallery() -> TData:
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]).long()
    query_mask = torch.ones_like(labels)
    gallery_mask = torch.ones_like(labels)

    input_tensors = labels.clone()
    input_tensors[0] = 3

    cmc_gt = 7 / 8

    return labels, query_mask, gallery_mask, input_tensors, cmc_gt


class DummyQGDataset(IDatasetQueryGallery):
    def __init__(self, labels: Tensor, gallery_mask: Tensor, query_mask: Tensor, input_tensors: Tensor):
        assert len(labels) == len(gallery_mask) == len(query_mask)

        self.labels = labels
        self.gallery_mask = gallery_mask
        self.query_mask = query_mask
        self.input_tensors = input_tensors

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            LABELS_KEY: self.labels[idx],
            INPUT_TENSORS_KEY: self.input_tensors[idx],
            IS_QUERY_KEY: self.query_mask[idx],
            IS_GALLERY_KEY: self.gallery_mask[idx],
        }

    def __len__(self) -> int:
        return len(self.labels)


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("data", [get_separate_query_gallery(), get_shared_query_gallery()])
def test_retrieval_validation(batch_size: int, shuffle: bool, num_workers: int, data: TData) -> None:
    labels, query_mask, gallery_mask, input_tensors, cmc_gt = data

    dataset = DummyQGDataset(
        labels=labels,
        input_tensors=input_tensors,
        query_mask=query_mask,
        gallery_mask=gallery_mask,
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
