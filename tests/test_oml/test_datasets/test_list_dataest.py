import torch

from oml.const import MOCK_DATASET_PATH
from oml.datasets.base import ListDataset


def test_dataset_len() -> None:
    images = list((MOCK_DATASET_PATH / "images").iterdir())
    dataset = ListDataset(images)

    assert len(dataset) == len(images)


def test_dataset_iter() -> None:
    images = list((MOCK_DATASET_PATH / "images").iterdir())
    dataset = ListDataset(images)

    for im in dataset:
        assert isinstance(im, torch.Tensor)
