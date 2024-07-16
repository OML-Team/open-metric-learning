import pandas as pd
import pytest
import torch

from oml.const import PATHS_COLUMN
from oml.datasets import AudioBaseDataset, AudioLabeledDataset
from tests.test_oml.test_datasets.test_datasets import get_df, get_df_with_start_times


@pytest.mark.needs_optional_dependency
@pytest.mark.parametrize("df", (get_df(), get_df_with_start_times()))
@pytest.mark.parametrize("convert_to_mono", [True, False])
def test_downmix(df: pd.DataFrame, convert_to_mono: bool) -> None:
    dataset = AudioBaseDataset(df[PATHS_COLUMN].tolist(), convert_to_mono=convert_to_mono)
    for item in dataset:
        audio = item[dataset.input_tensors_key]
        if convert_to_mono:
            assert audio.shape[0] == 1, f"Audio should be mono, but has {audio.shape[0]} channels"


@pytest.mark.needs_optional_dependency
@pytest.mark.parametrize("df", (get_df(), get_df_with_start_times()))
@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100])
@pytest.mark.parametrize("max_num_seconds", [0.01, 3.0, 100.0])
def test_resample_trim_pad(df: pd.DataFrame, sample_rate: int, max_num_seconds: float) -> None:
    dataset = AudioBaseDataset(df[PATHS_COLUMN].tolist(), sample_rate=sample_rate, max_num_seconds=max_num_seconds)
    for item in dataset:
        audio = item[dataset.input_tensors_key]
        assert audio.shape[1] == int(
            max_num_seconds * sample_rate
        ), f"Audio length {audio.shape[1]} does not match expected {int(max_num_seconds * sample_rate)}"


@pytest.mark.needs_optional_dependency
def test_start_times() -> None:
    df = get_df_with_start_times()
    dataset = AudioLabeledDataset(df)
    for _ in dataset:
        pass
    assert True, "Dataset iteration failed with start times"


@pytest.mark.needs_optional_dependency
def test_trim_or_pad_error() -> None:
    audio_tensor = torch.randn(1, 1000, dtype=torch.float)
    frame_offset = 1200
    num_frames = 500

    with pytest.raises(
        ValueError, match=f"The frame offset {frame_offset} is greater than the audio length {audio_tensor.shape[1]}."
    ):
        AudioBaseDataset._trim_or_pad(audio=audio_tensor, frame_offset=frame_offset, num_frames=num_frames)
