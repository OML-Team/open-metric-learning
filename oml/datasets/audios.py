from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import FloatTensor

from oml.const import (
    AUDIO_EXTENSIONS,
    BLACK,
    DEFAULT_CONVERT_TO_MONO,
    DEFAULT_SAMPLE_RATE,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    LABELS_KEY,
    MAX_AUDIO_LEN,
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
        df[START_TIME_COLUMN] = df[START_TIME_COLUMN].fillna(0.0)
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
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = MAX_AUDIO_LEN,
        convert_to_mono: bool = DEFAULT_CONVERT_TO_MONO,
        start_times: Optional[List[float]] = None,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """

        Args:
            paths: List of audio file paths.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            sample_rate: Sampling rate of audio files.
            max_num_seconds: Duration to use for each audio file.
            convert_to_mono: Whether to downmix audio to one channel or leave the same.
            start_times: List of start time offsets in ``seconds`` for each audio.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        assert (start_times is None) or (
            len(paths) == len(start_times)
        ), "The length of 'start_times' must match the length of 'paths' if 'start_times' is provided."
        assert sample_rate > 0, "The sample rate must be a positive integer."
        assert max_num_seconds > 0, "The maximum number of seconds must be a positive float."
        assert isinstance(convert_to_mono, bool), "'convert_to_mono' must be a boolean."

        paths = [Path(p) if dataset_root is None else Path(dataset_root) / p for p in paths]
        assert all(
            path.suffix in AUDIO_EXTENSIONS for path in paths
        ), f"Input audios should have one of '{AUDIO_EXTENSIONS}' extensions."

        if extra_data is not None:
            assert all(
                len(record) == len(paths) for record in extra_data.values()
            ), "All the extra records need to have the size equal to the dataset's size"
            self.extra_data = extra_data
        else:
            self.extra_data = {}

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

        self._paths = paths
        self._sample_rate = sample_rate
        self._num_frames = int(max_num_seconds * sample_rate)
        self._convert_to_mono = convert_to_mono
        self._frame_offsets = (
            [int(st * sample_rate) for st in start_times] if start_times is not None else [0] * len(paths)
        )
        self._spectral_function = spec_repr_func or default_spec_repr_func

    def _downmix_and_resample(self, audio: FloatTensor, sample_rate: int) -> FloatTensor:
        """
        (Optionally) downmix audio to mono and resample it to the dataset's sampling rate.

        Args:
            audio: Input audio tensor.
            sample_rate: Original sampling rate of the audio.

        Returns:
            Processed audio tensor.
        """
        from torchaudio.transforms import Resample

        if self._convert_to_mono and audio.shape[0] != 1:
            audio = audio.mean(dim=1, keepdim=True)
        if sample_rate != self._sample_rate:
            resampler = Resample(sample_rate, self._sample_rate)
            audio = resampler(audio)
        return audio

    @staticmethod
    def _trim_or_pad(audio: FloatTensor, frame_offset: int, num_frames: int) -> FloatTensor:
        """
        Trim and/or pad the audio to match the desired number of frames.

        Args:
            audio: Audio tensor.
            frame_offset: Frame offset for trimming the audio tensor.
            num_frames: Desired number of frames to be in the audio tensor.

        Returns:
            Trimmed and/or padded audio tensor.
        """
        if audio.shape[1] < frame_offset:
            raise ValueError(f"The frame offset {frame_offset} is greater than the audio length {audio.shape[1]}.")
        if audio.shape[1] > num_frames:
            audio = audio[:, frame_offset : frame_offset + num_frames]
        if audio.shape[1] < num_frames:
            padding = (num_frames - audio.shape[1], 0)
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
        audio, sample_rate = torchaudio.load(path)
        audio = self._downmix_and_resample(audio, sample_rate)
        audio = self._trim_or_pad(audio, self._frame_offsets[item], self._num_frames)
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
        spec_repr = self._spectral_function(audio)
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
        spec_repr = self._spectral_function(audio)
        return visualize_audio_with_player(audio=audio, spec_repr=spec_repr, sample_rate=self._sample_rate, color=color)


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
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = MAX_AUDIO_LEN,
        convert_to_mono: bool = DEFAULT_CONVERT_TO_MONO,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """

        Args:
            df: DataFrame with input data.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            labels_key: Key under which labels are stored.
            sample_rate: Sampling rate of audio files.
            max_num_seconds: Duration to use from each audio file.
            convert_to_mono: Whether to downmix audio to one channel or leave the same.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        dataset = AudioBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            sample_rate=sample_rate,
            max_num_seconds=max_num_seconds,
            convert_to_mono=convert_to_mono,
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
    The `non-annotated` dataset of audios having `query`/`gallery` split.
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
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = MAX_AUDIO_LEN,
        convert_to_mono: bool = DEFAULT_CONVERT_TO_MONO,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """

        Args:
            df: DataFrame with input data.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            sample_rate: Sampling rate of audio files.
            max_num_seconds: Duration to use from each audio file.
            convert_to_mono: Whether to downmix audio to one channel or leave the same.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        dataset = AudioBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            sample_rate=sample_rate,
            max_num_seconds=max_num_seconds,
            convert_to_mono=convert_to_mono,
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
    The `annotated` dataset of audios having `query`/`gallery` split.
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
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_num_seconds: float = MAX_AUDIO_LEN,
        convert_to_mono: bool = DEFAULT_CONVERT_TO_MONO,
        spec_repr_func: Callable[[FloatTensor], FloatTensor] = default_spec_repr_func,
    ):
        """

        Args:
            df: DataFrame with input data.
            dataset_root: Base path for audio files.
            extra_data: Extra data to include in dataset items.
            input_tensors_key: Key under which audio tensors are stored.
            index_key: Key for indexing dataset items.
            labels_key: Key under which labels are stored.
            sample_rate: Sampling rate of audio files.
            max_num_seconds: Duration to use from each audio file.
            convert_to_mono: Whether to downmix audio to one channel or leave the same.
            spec_repr_func: Spectral representation extraction function used for visualization.
        """
        dataset = AudioBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            sample_rate=sample_rate,
            max_num_seconds=max_num_seconds,
            convert_to_mono=convert_to_mono,
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
