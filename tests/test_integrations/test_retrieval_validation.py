from functools import partial
from math import isclose
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from oml.const import EMBEDDINGS_KEY, INPUT_TENSORS_KEY, OVERALL_CATEGORIES_KEY
from oml.datasets import EmbeddingsQueryGalleryDataset
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import one_hot
from tests.test_integrations.utils import IdealClusterEncoder

oh = partial(one_hot, dim=8)


def get_separate_query_gallery() -> Any:
    dataset = EmbeddingsQueryGalleryDataset(
        labels=torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 1]).long(),
        is_gallery=torch.tensor([0, 1, 0, 1, 0, 0, 1, 1, 0]).bool(),
        is_query=torch.tensor([1, 0, 1, 0, 1, 1, 0, 0, 1]).bool(),
        # let's add some errors on positions 4,5 in queries
        embeddings=torch.tensor([0, 0, 1, 1, 0, 0, 2, 2, 1]).float(),
    )

    cmc_gt = 3 / 5

    return dataset, cmc_gt


def get_shared_query_gallery() -> Any:
    dataset = EmbeddingsQueryGalleryDataset(
        labels=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]).long(),
        is_query=torch.ones(8).bool(),
        is_gallery=torch.ones(8).bool(),
        embeddings=torch.tensor([3, 0, 0, 1, 1, 1, 2, 2]).float(),
    )

    cmc_gt = 7 / 8

    return dataset, cmc_gt


@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("shuffle", [False])  # todo 522: supper True?
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("data", [get_separate_query_gallery(), get_shared_query_gallery()])
def test_retrieval_validation(batch_size, shuffle, num_workers, data) -> None:  # type: ignore
    dataset, cmc_gt = data

    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers
    )

    calc = EmbeddingMetrics(cmc_top_k=(1,), dataset=dataset)
    calc.setup(num_samples=len(dataset))

    model = IdealClusterEncoder()

    for batch in loader:
        output = model(batch[INPUT_TENSORS_KEY])
        batch[EMBEDDINGS_KEY] = output
        calc.update_data(embeddings=batch[EMBEDDINGS_KEY])

    metrics = calc.compute_metrics()

    assert isclose(metrics[OVERALL_CATEGORIES_KEY]["cmc"][1], cmc_gt, abs_tol=1e-5)
