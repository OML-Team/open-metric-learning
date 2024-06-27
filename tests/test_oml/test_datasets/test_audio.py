import random

import pandas as pd
import pytest

from oml.const import PATHS_COLUMN, START_TIME_COLUMN
from oml.datasets.audios import (
    AudioBaseDataset,
    AudioLabeledDataset,
    AudioQueryGalleryDataset,
    AudioQueryGalleryLabeledDataset,
)
from oml.utils import get_mock_audios_dataset
from tests.test_oml.test_datasets.test_datasets import (
    check_base,
    check_labeled,
    check_query_gallery,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    df_train, df_val = get_mock_audios_dataset(global_paths=True)
    return pd.concat([df_train, df_val])


@pytest.mark.needs_optional_dependency
@pytest.mark.parametrize("num_channels", [1, 2])
def test_downmix(df: pd.DataFrame, num_channels: int) -> None:
    dataset = AudioBaseDataset(df[PATHS_COLUMN].tolist(), num_channels=num_channels)
    for item in dataset:
        audio = item[dataset.input_tensors_key]
        assert audio.shape[0] <= num_channels, f"Audio channels {audio.shape[0]} exceed specified {num_channels}"


@pytest.mark.needs_optional_dependency
@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("max_num_seconds", [0.01, 0.5, 1.0, 3.0, 10.0, 100.0])
def test_resample_trim_pad(df: pd.DataFrame, sr: int, max_num_seconds: float) -> None:
    dataset = AudioBaseDataset(df[PATHS_COLUMN].tolist(), sr=sr, max_num_seconds=max_num_seconds)
    for item in dataset:
        audio = item[dataset.input_tensors_key]
        assert audio.shape[1] == int(
            max_num_seconds * sr
        ), f"Audio length {audio.shape[1]} does not match expected {int(max_num_seconds * sr)}"


@pytest.mark.needs_optional_dependency
def test_start_times(df: pd.DataFrame) -> None:
    dataset = AudioBaseDataset(df[PATHS_COLUMN].tolist(), use_random_start=False)
    for _ in dataset:
        pass
    assert True, "Dataset iteration failed without random start"

    dataset = AudioBaseDataset(df[PATHS_COLUMN].tolist(), use_random_start=True)
    for _ in dataset:
        pass
    assert True, "Dataset iteration failed with random start"

    extra_data = {START_TIME_COLUMN: [random.uniform(0, 1) for _ in range(len(df))]}
    dataset = AudioBaseDataset(df[PATHS_COLUMN].tolist(), use_random_start=True, extra_data=extra_data)
    for _ in dataset:
        pass
    assert True, "Dataset iteration failed with start times"


@pytest.mark.needs_optional_dependency
def test_audio_datasets() -> None:
    df_train, df_val = get_mock_audios_dataset(global_paths=True)
    df = pd.concat([df_train, df_val])

    # Base
    dataset_b = AudioBaseDataset(paths=df[PATHS_COLUMN].tolist())
    check_base(dataset_b)

    # Labeled
    dataset_l = AudioLabeledDataset(df_train)
    check_base(dataset_l)
    check_labeled(dataset_l, df_train)

    # Query Gallery
    dataset_qg = AudioQueryGalleryDataset(df_val)
    check_base(dataset_qg)
    check_query_gallery(dataset_qg, df_val)

    # Query Gallery Labeled
    dataset_qgl = AudioQueryGalleryLabeledDataset(df_val)
    check_base(dataset_qgl)
    check_query_gallery(dataset_qgl, df_val)
    check_labeled(dataset_qgl, df_val)
