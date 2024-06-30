from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import FloatTensor

from oml.const import (
    AUDIO_EXTENSIONS,
    BLACK,
    DEFAULT_AUDIO_NUM_CHANNELS,
    DEFAULT_DURATION,
    DEFAULT_SAMPLE_RATE,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    LABELS_KEY,
    PATHS_COLUMN,
    START_TIME_COLUMN,
    TColor,
)
from oml.datasets.dataframe import (
    DFLabeledDataset,
    DFQueryGalleryDataset,
    DFQueryGalleryLabeledDataset,
)
from oml.interfaces.datasets import IBaseDataset, IVisualizableDataset
from oml.utils.audios import (
    default_spec_repr_func,
    visualize_audio,
    visualize_audio_with_player,
)


def parse_start_times(df: pd.DataFrame) -> Optional[List[float]]:
    """
    Parses starting time points from DataFrame.
    """
    start_times = None
    if START_TIME_COLUMN in df.columns:
        assert (
            df[START_TIME_COLUMN].dtype == float
        ), f"Expected dtype of '{START_TIME_COLUMN}' column is `float`, `{df[START_TIME_COLUMN].dtype}` is found."
        start_times = df[START_TIME_COLUMN].astype(float).tolist()
    return start_times


class AudioBaseDataset(IBaseDataset, IVisualizableDataset):
    """
    The base class that handles audio specific logic.
    """

    def __init__(
        self,
        paths: List[Union[str, Path]],
        dataset_root: Optional[Union[str, Path]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
        sr: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_AUDIO_NUM_CHANNELS,
        start_times: Optional[List[float]] = None,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """
        Initializes the AudioDataset.

        Args:
            paths: List of audio file paths.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            sr: Sampling rate of audio files.
            max_num_seconds: Duration to use from each audio file.
            num_channels: Number of audio channels.
            start_times: List of start time offsets in **seconds** for each audio.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        assert (start_times is None) or (len(paths) == len(start_times))

        if extra_data is not None:
            assert all(
                len(record) == len(paths) for record in extra_data.values()
            ), "All the extra records need to have the size equal to the dataset's size"
            self.extra_data = extra_data
        else:
            self.extra_data = {}

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

        self._paths = [Path(p) if dataset_root is None else Path(dataset_root) / p for p in paths]
        assert all(
            path.suffix in AUDIO_EXTENSIONS for path in self._paths
        ), f"Input audios should have one of '{AUDIO_EXTENSIONS}' extensions."

        self.sr = sr
        self.num_frames = int(max_num_seconds * sr)
        self.num_channels = num_channels
        self.frame_offsets = [int(st * sr) for st in start_times] if start_times is not None else [0] * len(paths)
        self.spectral_function = spec_repr_func or default_spec_repr_func

    def _downmix_and_resample(self, audio: FloatTensor, sr: int) -> FloatTensor:
        """
        Downmix audio to mono and resample it to the dataset's sampling rate.

        Args:
            audio: Input audio tensor.
            sr: Original sampling rate of the audio.

        Returns:
            Processed audio tensor.
        """
        from torchaudio.transforms import Resample

        if audio.shape[0] != self.num_channels:
            audio = audio.mean(dim=1, keepdim=True)
        if sr != self.sr:
            resampler = Resample(sr, self.sr)
            audio = resampler(audio)
        return audio

    def _trim_or_pad(self, audio: FloatTensor, frame_offset: int) -> FloatTensor:
        """
        Trim or pad the audio to match the desired number of frames.
        if `start_time` is specified, then it will be used, else if

        Args:
            audio: Audio tensor.
            frame_offset: Frame offset for trimming.

        Returns:
            Trimmed or padded audio tensor.
        """
        audio_length = audio.shape[1]
        if audio_length > self.num_frames:
            audio = audio[:, frame_offset : frame_offset + self.num_frames]
        else:
            padding = (self.num_frames - audio_length, 0)
            audio = torch.nn.functional.pad(audio, padding)
        return audio

    def get_audio(self, item: int) -> FloatTensor:
        """
        Load and process an audio file.

        Args:
            item: Dataset item index.

        Returns:
            Processed audio tensor.
        """
        import torchaudio

        path = self._paths[item]
        audio, sr = torchaudio.load(path)
        audio = self._downmix_and_resample(audio, sr)
        audio = self._trim_or_pad(audio, self.frame_offsets[item])
        return audio

    def __getitem__(self, item: int) -> Dict[str, Union[FloatTensor, int]]:
        audio_tensor = self.get_audio(item)
        data = {
            self.input_tensors_key: audio_tensor,
            self.index_key: item,
        }
        for key, record in self.extra_data.items():
            if key in data:
                raise ValueError(f"<extra_data> and dataset share the same key: {key}")
            else:
                data[key] = record[item]

        return data

    def __len__(self) -> int:
        return len(self._paths)

    def visualize(self, item: int, color: TColor = BLACK) -> np.ndarray:
        """
        Visualize an audio file.

        Args:
            item: Dataset item index.
            color: Color of the plot.

        Returns:
            Array representing the image of the plot.
        """
        audio = self.get_audio(item)
        spec_repr = self.spectral_function(audio)
        return visualize_audio(spec_repr=spec_repr, color=color)

    def visualize_with_player(self, item: int, color: TColor = BLACK) -> str:
        """
        Visualize an audio file in HTML markup.

        Args:
            item: Dataset item index.
            color: Color of the plot.

        Returns:
            HTML markup with spectral representation image and audio player.
        """
        audio = self.get_audio(item)
        spec_repr = self.spectral_function(audio)
        return visualize_audio_with_player(audio=audio, spec_repr=spec_repr, sr=self.sr, color=color)


class AudioLabeledDataset(DFLabeledDataset, IVisualizableDataset):
    """
    The dataset of audios having their ground truth labels.

    """

    _dataset: AudioBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Optional[Union[str, Path]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
        labels_key: str = LABELS_KEY,
        sr: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_AUDIO_NUM_CHANNELS,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """
        Initializes the AudioLabeledDataset.

        Args:
            df: DataFrame with input data.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            labels_key: Key under which labels are stored.
            sr: Sampling rate of audio files.
            max_num_seconds: Duration to use from each audio file.
            num_channels: Number of audio channels.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        dataset = AudioBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            sr=sr,
            max_num_seconds=max_num_seconds,
            num_channels=num_channels,
            start_times=parse_start_times(df),
            spec_repr_func=spec_repr_func,
        )
        super().__init__(dataset=dataset, df=df, extra_data=extra_data, labels_key=labels_key)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)

    def visualize_with_player(self, item: int, color: TColor) -> str:
        return self._dataset.visualize_with_player(item=item, color=color)


class AudioQueryGalleryDataset(DFQueryGalleryDataset, IVisualizableDataset):
    """
    The **non-annotated** dataset of audios having `query`/`gallery` split.
    To perform `1 vs rest` validation, where a query is evaluated versus the whole validation dataset
    (except for this exact query), you should mark the item as ``is_query == True`` and ``is_gallery == True``.

    """

    _dataset: AudioBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Optional[Union[str, Path]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
        sr: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_AUDIO_NUM_CHANNELS,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """
        Initializes the AudioQueryGalleryDataset.

        Args:
            df: DataFrame with input data.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            sr: Sampling rate of audio files.
            max_num_seconds: Duration to use from each audio file.
            num_channels: Number of audio channels.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        dataset = AudioBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            sr=sr,
            max_num_seconds=max_num_seconds,
            num_channels=num_channels,
            start_times=parse_start_times(df),
            spec_repr_func=spec_repr_func,
        )
        super().__init__(dataset=dataset, df=df, extra_data=extra_data)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)

    def visualize_with_player(self, item: int, color: TColor) -> str:
        return self._dataset.visualize_with_player(item=item, color=color)


class AudioQueryGalleryLabeledDataset(DFQueryGalleryLabeledDataset, IVisualizableDataset):
    """
    The **annotated** dataset of audios having `query`/`gallery` split.
    To perform `1 vs rest` validation, where a query is evaluated versus the whole validation dataset
    (except for this exact query), you should mark the item as ``is_query == True`` and ``is_gallery == True``.

    """

    _dataset: AudioBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Optional[Union[str, Path]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
        labels_key: str = LABELS_KEY,
        sr: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_AUDIO_NUM_CHANNELS,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """
        Initializes the AudioQueryGalleryLabeledDataset.

        Args:
            df: DataFrame with input data.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            labels_key: Key under which labels are stored.
            sr: Sampling rate of audio files.
            max_num_seconds: Duration to use from each audio file.
            num_channels: Number of audio channels.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        dataset = AudioBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            sr=sr,
            max_num_seconds=max_num_seconds,
            num_channels=num_channels,
            start_times=parse_start_times(df),
            spec_repr_func=spec_repr_func,
        )
        super().__init__(dataset=dataset, df=df, extra_data=extra_data, labels_key=labels_key)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)

    def visualize_with_player(self, item: int, color: TColor) -> str:
        return self._dataset.visualize_with_player(item=item, color=color)


__all__ = [
    "AudioBaseDataset",
    "AudioLabeledDataset",
    "AudioQueryGalleryDataset",
    "AudioQueryGalleryLabeledDataset",
]
