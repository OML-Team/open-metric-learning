from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from oml.const import MOCK_DATASET_PATH
from oml.datasets.retrieval import DatasetWithLabels
from oml.registry.transforms import AUGS_REGISTRY, get_augs
from oml.utils.images.images import imread_cv2, imread_pillow


@pytest.mark.parametrize("aug_name", [None, *list(AUGS_REGISTRY.keys())])
@pytest.mark.parametrize("f_imread", [imread_cv2, imread_pillow])
def test_transforms(aug_name: Optional[str], f_imread: Any) -> None:
    transforms = get_augs(name=aug_name) if aug_name is not None else None

    df = pd.read_csv(MOCK_DATASET_PATH / "df.csv")
    df["path"] = df["path"].apply(Path)

    dataset = DatasetWithLabels(
        df=df, im_size=32, pad_ratio=0, dataset_root=MOCK_DATASET_PATH, transform=transforms, f_imread=f_imread
    )

    _ = dataset[0]

    assert True
