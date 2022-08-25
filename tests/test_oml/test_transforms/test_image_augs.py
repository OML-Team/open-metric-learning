from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from oml.const import MOCK_DATASET_PATH
from oml.datasets.retrieval import DatasetWithLabels
from oml.registry.transforms import TRANSFORMS_REGISTRY, get_transforms
from oml.transforms.images.utils import get_im_reader_for_transforms


@pytest.mark.parametrize("aug_name", [None, *list(TRANSFORMS_REGISTRY.keys())])
def test_transforms(aug_name: Optional[str]) -> None:
    transforms = get_transforms(name=aug_name, im_size=128) if aug_name else None  # type: ignore

    f_imread = get_im_reader_for_transforms(transforms)

    df = pd.read_csv(MOCK_DATASET_PATH / "df.csv")
    df["path"] = df["path"].apply(Path)

    dataset = DatasetWithLabels(df=df, dataset_root=MOCK_DATASET_PATH, transform=transforms, f_imread=f_imread)

    _ = dataset[0]

    assert True
