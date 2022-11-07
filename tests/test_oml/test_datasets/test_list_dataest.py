from pathlib import Path
from typing import Iterator, List, Tuple

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from oml.const import MOCK_DATASET_PATH
from oml.datasets.list_ import ListDataset


@pytest.fixture
def images() -> Iterator[List[Path]]:
    yield list((MOCK_DATASET_PATH / "images").iterdir())


def get_images_and_boxes() -> List[Tuple[Path, int, int, int, int]]:
    df = pd.read_csv(MOCK_DATASET_PATH / "df_with_bboxes.csv")
    sub_df = df[["path", "x_1", "y_1", "x_2", "y_2"]]
    sub_df["path"] = sub_df.path.apply(lambda p: MOCK_DATASET_PATH / p)
    lines: List[Tuple[Path, int, int, int, int]] = [tuple(line[1]) for line in sub_df.iterrows()]  # type: ignore
    return lines


def test_dataset_len(images: List[Path]) -> None:
    assert len(images) > 0
    dataset = ListDataset(images)

    assert len(dataset) == len(images)


def test_dataset_iter(images: List[Path]) -> None:
    dataset = ListDataset(images)

    for im in dataset:
        assert isinstance(im, torch.Tensor)


def test_dataloader_iter(images: List[Path]) -> None:
    dataloader = DataLoader(ListDataset(images))

    for im in dataloader:
        assert im.ndim == 4


@pytest.mark.parametrize("im_path,x1,y1,x2,y2", get_images_and_boxes())
def test_mock_dataset_iter(im_path: Path, x1: int, x2: int, y1: int, y2: int) -> None:
    dataloader = DataLoader(ListDataset([im_path], [(x1, y1, x2, y2)]))

    image = next(iter(dataloader))
    assert image.size() == (1, 3, x2 - x1, y2 - y1)


def test_mock_dataset_iter_with_nones() -> None:
    import random

    random.seed(42)
    paths = []
    bboxes = []
    for row in get_images_and_boxes():
        paths.append(row[0])
        bboxes.append((row[1], row[2], row[3], row[4]))
        if random.random() > 0.5:
            paths.append(row[0])
            bboxes.append(None)

    dataloader = DataLoader(ListDataset(paths, bboxes))
    for image in dataloader:
        assert image.ndim == 4
