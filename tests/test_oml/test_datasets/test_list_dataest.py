import csv

import torch
import pytest
from torch.utils.data import DataLoader

from oml.const import MOCK_DATASET_PATH
from oml.datasets.list_ import ListDataset


@pytest.fixture
def images():
    yield list((MOCK_DATASET_PATH / "images").iterdir())

def get_images_and_boxes():
    lines = []
    with (MOCK_DATASET_PATH / "df_with_bboxes.csv").open() as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            im_path = line[1]
            x1, x2, y1, y2 = map(int, line[-4:])
            lines.append((MOCK_DATASET_PATH / im_path, x1, x2, y1, y2))

    return lines


def test_dataset_len(images) -> None:
    assert len(images) > 0
    dataset = ListDataset(images)

    assert len(dataset) == len(images)


def test_dataset_iter(images) -> None:
    dataset = ListDataset(images)

    for im in dataset:
        assert isinstance(im, torch.Tensor)


def test_dataloader_iter(images) -> None:
    dataloader = DataLoader(ListDataset(images))

    for im in dataloader:
        assert im.ndim == 4


@pytest.mark.parametrize("im_path,x1,x2,y1,y2", get_images_and_boxes())
def test_mock_dataset_iter(im_path, x1, x2, y1, y2):
    dataloader = DataLoader(ListDataset([im_path], {0: (x1, y1, x2, y2)}))

    image = next(iter(dataloader))
    assert image.size() == (1, 3, x2-x1, y2-y1)
