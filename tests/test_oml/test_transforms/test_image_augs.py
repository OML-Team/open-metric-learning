from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from oml.const import MOCK_DATASET_PATH
from oml.datasets.retrieval import DatasetWithLabels
from oml.registry.transforms import TRANSFORMS_REGISTRY, get_transforms
from oml.utils.images.images import imread_cv2, imread_pillow


@pytest.mark.parametrize("aug_name", [None, *list(TRANSFORMS_REGISTRY.keys())])
@pytest.mark.parametrize("f_imread", [imread_cv2, imread_pillow])
def test_transforms(aug_name: Optional[str], f_imread: Any) -> None:
    transforms = get_transforms(name=aug_name, im_size=128) if aug_name else None  # type: ignore

    df = pd.read_csv(MOCK_DATASET_PATH / "df.csv")
    df["path"] = df["path"].apply(Path)

    dataset = DatasetWithLabels(df=df, dataset_root=MOCK_DATASET_PATH, transform=transforms, f_imread=f_imread)

    _ = dataset[0]

    assert True
