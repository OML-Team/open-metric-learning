from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from omegaconf import OmegaConf

from oml.const import CONFIGS_PATH, MOCK_DATASET_PATH
from oml.datasets.retrieval import DatasetWithLabels
from oml.registry.transforms import TRANSFORMS_REGISTRY, get_transforms_by_cfg
from oml.transforms.images.utils import get_im_reader_for_transforms


@pytest.mark.parametrize("aug_name", list(TRANSFORMS_REGISTRY.keys()))
def test_transforms(aug_name: Optional[str]) -> None:
    path_to_cfg = CONFIGS_PATH / "transforms" / f"{aug_name}.yaml"
    print(path_to_cfg)
    with open(path_to_cfg, "r") as f:
        cfg = OmegaConf.load(f)

    transforms = get_transforms_by_cfg(cfg)

    f_imread = get_im_reader_for_transforms(transforms)

    df = pd.read_csv(MOCK_DATASET_PATH / "df.csv")
    df["path"] = df["path"].apply(Path)

    dataset = DatasetWithLabels(df=df, dataset_root=MOCK_DATASET_PATH, transform=transforms, f_imread=f_imread)

    _ = dataset[0]

    assert True
