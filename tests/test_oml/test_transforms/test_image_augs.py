from typing import Optional

import pandas as pd
import pytest
from omegaconf import OmegaConf

from oml.const import CONFIGS_PATH, MOCK_DATASET_PATH
from oml.datasets import ImagesDatasetWithLabels
from oml.registry.transforms import TRANSFORMS_REGISTRY, get_transforms_by_cfg


@pytest.mark.parametrize("aug_name", list(TRANSFORMS_REGISTRY.keys()))
def test_transforms(aug_name: Optional[str]) -> None:
    df = pd.read_csv(MOCK_DATASET_PATH / "df.csv")
    transforms = get_transforms_by_cfg(OmegaConf.load(CONFIGS_PATH / "transforms" / f"{aug_name}.yaml"))

    dataset = ImagesDatasetWithLabels(df=df, dataset_root=MOCK_DATASET_PATH, transform=transforms)

    _ = dataset[0]

    assert True


def test_default_transforms() -> None:
    df = pd.read_csv(MOCK_DATASET_PATH / "df.csv")
    dataset = ImagesDatasetWithLabels(df=df, dataset_root=MOCK_DATASET_PATH, transform=None)
    _ = dataset[0]
    assert True
