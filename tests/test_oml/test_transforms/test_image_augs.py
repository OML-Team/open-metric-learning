from typing import Optional

import pandas as pd
import pytest

from oml.datasets.retrieval import DatasetWithLabels
from oml.registry.transforms import get_augs
from tests.conftest import TESTS_MOCK_DATASET


@pytest.mark.parametrize("aug_name", [None, "default_albu", "default_weak_albu", "default_torch"])
def test(aug_name: Optional[str]) -> None:
    transforms = get_augs(name=aug_name) if aug_name is not None else None

    df = pd.read_csv(TESTS_MOCK_DATASET / "df.csv")
    dataset = DatasetWithLabels(
        df=df, im_size=32, pad_ratio=0, images_root=TESTS_MOCK_DATASET / "images", transform=transforms
    )

    _ = dataset[0]

    assert True
