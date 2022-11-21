from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from oml.const import MOCK_DATASET_PATH
from oml.datasets.list_ import ListDataset, TBBox


@pytest.fixture
def images() -> Iterator[List[Path]]:
    yield list((MOCK_DATASET_PATH / "images").iterdir())


def get_images_and_boxes() -> Tuple[List[Path], List[TBBox]]:
    df = pd.read_csv(MOCK_DATASET_PATH / "df_with_bboxes.csv")
    sub_df = df[["path", "x_1", "y_1", "x_2", "y_2"]]
    sub_df["path"] = sub_df["path"].apply(lambda p: MOCK_DATASET_PATH / p)
    paths, bboxes = [], []
    for row in sub_df.iterrows():
        path, x1, y1, x2, y2 = row[1]
        paths.append(path)
        bboxes.append((x1, y1, x2, y2))
    return paths, bboxes


def get_images_and_boxes_with_nones() -> Tuple[List[Path], List[Optional[TBBox]]]:
    import random

    random.seed(42)

    paths, bboxes = [], []
    for path, bbox in zip(*get_images_and_boxes()):
        paths.append(path)
        bboxes.append(bbox)
        if random.random() > 0.5:
            paths.append(path)
            bboxes.append(None)
    return paths, bboxes


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


@pytest.mark.parametrize("im_paths,bboxes", [get_images_and_boxes(), get_images_and_boxes_with_nones()])
def test_list_dataset_iter(im_paths: Sequence[Path], bboxes: Sequence[Optional[TBBox]]) -> None:
    dataloader = DataLoader(ListDataset(im_paths, bboxes))
    for image, box in zip(dataloader, bboxes):
        if box is not None:
            x1, y1, x2, y2 = box
        else:
            x1, y1, x2, y2 = 0, 0, image.size()[2], image.size()[3]
        assert image.ndim == 4
        assert image.size() == (1, 3, x2 - x1, y2 - y1)
