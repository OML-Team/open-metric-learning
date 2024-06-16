from pathlib import Path
from typing import List

import pytest
import torch
from torchvision.models import resnet18

from oml.const import MOCK_DATASET_PATH
from oml.datasets import ImageBaseDataset
from oml.inference import inference
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.utils.download_mock_dataset import download_mock_dataset
from tests.test_oml.test_ddp.utils import init_ddp, run_in_ddp


@pytest.mark.long
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("world_size,n_paths", [(3, 5), (3, 6)])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_inference_without_expected_duplicates(world_size: int, n_paths: int, device: str, batch_size: int) -> None:
    _, df_val = download_mock_dataset(MOCK_DATASET_PATH)
    paths = df_val["path"].apply(lambda x: MOCK_DATASET_PATH / x).tolist()

    assert len(paths) >= n_paths, "Something is wrong with mock dataset. You should fix it."
    paths = paths[:n_paths]

    run_in_ddp(world_size=world_size, fn=run_with_handling_duplicates, args=(device, paths, batch_size))


def run_with_handling_duplicates(rank: int, world_size: int, device: str, paths: List[str], batch_size: int) -> None:
    init_ddp(rank, world_size)

    transform = get_normalisation_resize_torch(im_size=32)

    model = resnet18().eval().to(device)

    args = {
        "model": model,
        "dataset": ImageBaseDataset(paths=[Path(x) for x in paths], transform=transform),
        "num_workers": 0,
        "verbose": True,
        "batch_size": batch_size,
    }

    output = inference(**args)
    assert len(paths) == len(output), (len(paths), len(output))