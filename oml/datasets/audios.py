import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from numpy.typing import NDArray
from torch import BoolTensor, FloatTensor
from torchaudio.transforms import MelSpectrogram, Resample

from oml.const import (
    AUDIO_EXTENSIONS,
    BLACK,
    CATEGORIES_COLUMN,
    DEFAULT_AUDIO_NUM_CHANNELS,
    DEFAULT_DURATION,
    DEFAULT_MELSPEC_PARAMS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_USE_RANDOM_START,
    FRAME_OFFSET_COLUMN,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    LABELS_KEY,
    PATHS_COLUMN,
    SEQUENCE_COLUMN,
    TColor,
)
from oml.interfaces.datasets import (
    IBaseDataset,
    ILabeledDataset,
    IQueryGalleryDataset,
    IQueryGalleryLabeledDataset,
    IVisualizableDataset,
)
from oml.utils.audios import visualize_audio, visualize_audio_with_player


class AudioBaseDataset(IBaseDataset, IVisualizableDataset):
    """
    The base class that handles audio specific logic.

    Args:
        paths: List of audio file paths.
        dataset_root: Base path for audio files, optional.
        extra_data: Extra data to include in dataset items.
        input_tensors_key: Key under which audio tensors are stored.
        index_key : Key for indexing dataset items.
        sr: Sampling rate of audio files.
        max_num_seconds: Duration to use from each audio file.
        num_channels: Number of audio channels.
        use_random_start: Extract audio fragment randomly or use predefined frame offsets from `extra_data`.
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
        use_random_start: bool = DEFAULT_USE_RANDOM_START,
    ):
        """
        Initializes the AudioDataset.
        """
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
        self.use_random_start = use_random_start
        if not use_random_start:
            assert (
                FRAME_OFFSET_COLUMN in self.extra_data
            ), f"If `use_random_start` is False, `extra_data` must contain '{FRAME_OFFSET_COLUMN}'."
        self.frame_offsets: Optional[List[int]] = self.extra_data.get(FRAME_OFFSET_COLUMN)

    def _downmix_and_resample(self, audio: FloatTensor, sr: int) -> FloatTensor:
        """
        Downmix audio to mono and resample it to the dataset's sampling rate.

        Args:
            audio: Input audio tensor.
            sr: Original sampling rate of the audio.

        Returns:
            Processed audio tensor.
        """
        if audio.shape[0] != self.num_channels:
            audio = audio.mean(dim=1, keepdim=True)
        if sr != self.sr:
            resampler = Resample(sr, self.sr)
            audio = resampler(audio)
        return audio

    def _trim_or_pad(self, audio: FloatTensor, frame_offset: int) -> FloatTensor:
        """
        Trim or pad the audio to match the desired number of frames.

        Args:
            audio: Audio tensor.
            frame_offset: Starting frame offset for trimming.

        Returns:
            Trimmed or padded audio tensor.
        """
        audio_length = audio.shape[1]
        if audio_length > self.num_frames:
            if frame_offset is None:
                frame_offset = random.randrange(0, audio_length - self.num_frames)
            else:
                frame_offset = int(frame_offset * self.sr)
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
        path = self._paths[item]
        audio, sr = torchaudio.load(path)
        frame_offset = None if self.frame_offsets is None else self.frame_offsets[item]
        audio = self._downmix_and_resample(audio, sr)
        audio = self._trim_or_pad(audio, frame_offset)
        return audio

    def get_spectral_repr(self, audio: FloatTensor, params: Dict[str, Any] = DEFAULT_MELSPEC_PARAMS) -> FloatTensor:
        """
        Generate a spectral representation (by default, log-scaled MelSpec) from an audio signal.
        Used primarily for visualization.

        Parameters:
            audio: The input audio tensor.
            params: Parameters for spectral representation computation.

        Returns:
            The spectral representation of the input audio tensor.
        """
        melspectrogram = MelSpectrogram(sample_rate=self.sr, **params)
        melspec = melspectrogram(audio)
        log_melspec = torch.log1p(melspec).squeeze(0)
        return log_melspec

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

    def visualize(self, item: int, color: TColor = BLACK) -> NDArray[np.uint8]:
        """
        Visualize an audio file.

        Args:
            item: Dataset item index.
            color: Color of the plot.

        Returns:
            Array representing the image of the plot.
        """
        audio = self.get_audio(item)
        spec_repr = self.get_spectral_repr(audio).squeeze(0)
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
        spec_repr = self.get_spectral_repr(audio)
        return visualize_audio_with_player(audio=audio, spec_repr=spec_repr, sr=self.sr, color=color)


class AudioLabeledDataset(AudioBaseDataset, ILabeledDataset):
    """
    The dataset of audios having their ground truth labels.

    """

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
        use_random_start: bool = DEFAULT_USE_RANDOM_START,
    ):
        assert all(x in df.columns for x in (LABELS_COLUMN, PATHS_COLUMN))
        self.labels_key = labels_key
        self.df = df

        extra_data = extra_data or dict()

        if CATEGORIES_COLUMN in df.columns:
            extra_data[CATEGORIES_COLUMN] = df[CATEGORIES_COLUMN].copy()

        if SEQUENCE_COLUMN in df.columns:
            extra_data[SEQUENCE_COLUMN] = df[SEQUENCE_COLUMN].copy()

        if FRAME_OFFSET_COLUMN in df.columns:
            extra_data[FRAME_OFFSET_COLUMN] = df[FRAME_OFFSET_COLUMN].copy()

        super().__init__(
            paths=df[PATHS_COLUMN],
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            sr=sr,
            max_num_seconds=max_num_seconds,
            num_channels=num_channels,
            use_random_start=use_random_start,
        )
        self._labels = np.array(self.df[LABELS_COLUMN])
        self._label2category = (
            dict(zip(self.df[LABELS_COLUMN], self.df[CATEGORIES_COLUMN]))
            if CATEGORIES_COLUMN in self.df.columns
            else None
        )

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        data[self.labels_key] = self.df.iloc[item][LABELS_COLUMN]
        return data

    def get_labels(self) -> NDArray[np.int64]:
        return self._labels

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        return self._label2category


class AudioQueryGalleryLabeledDataset(AudioLabeledDataset, IQueryGalleryLabeledDataset):
    """
    The **annotated** dataset of audios having `query`/`gallery` split.
    To perform `1 vs rest` validation, where a query is evaluated versus the whole validation dataset
    (except for this exact query), you should mark the item as ``is_query == True`` and ``is_gallery == True``.

    """

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
        use_random_start: bool = DEFAULT_USE_RANDOM_START,
    ):
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN, LABELS_COLUMN, PATHS_COLUMN))
        self.df = df

        super().__init__(
            df=df,
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            labels_key=labels_key,
            sr=sr,
            max_num_seconds=max_num_seconds,
            num_channels=num_channels,
            use_random_start=use_random_start,
        )
        self.query_ids = BoolTensor(self.df[IS_QUERY_COLUMN]).nonzero().squeeze()
        self.gallery_ids = BoolTensor(self.df[IS_GALLERY_COLUMN]).nonzero().squeeze()

    def get_query_ids(self) -> BoolTensor:
        return self.query_ids

    def get_gallery_ids(self) -> BoolTensor:
        return self.gallery_ids


class AudioQueryGalleryDataset(IVisualizableDataset, IQueryGalleryDataset):
    """
    The **non-annotated** dataset of audios having `query`/`gallery` split.
    To perform `1 vs rest` validation, where a query is evaluated versus the whole validation dataset
    (except for this exact query), you should mark the item as ``is_query == True`` and ``is_gallery == True``.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Optional[Union[str, Path]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
        sr: int = DEFAULT_SAMPLE_RATE,
        num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_AUDIO_NUM_CHANNELS,
        use_random_start: bool = DEFAULT_USE_RANDOM_START,
    ):
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN, PATHS_COLUMN))
        df = deepcopy(df)
        df[LABELS_COLUMN] = "fake_label"

        self.__dataset = AudioQueryGalleryLabeledDataset(
            df=df,
            dataset_root=dataset_root,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            labels_key=LABELS_COLUMN,
            sr=sr,
            max_num_seconds=num_seconds,
            num_channels=num_channels,
            use_random_start=use_random_start,
        )

        self.extra_data = self.__dataset.extra_data
        self.input_tensors_key = self.__dataset.input_tensors_key
        self.index_key = self.__dataset.index_key

    def __getitem__(self, item: int) -> Dict[str, Any]:
        batch = self.__dataset[item]
        del batch[self.__dataset.labels_key]
        return batch

    def __len__(self) -> int:
        return len(self.__dataset)

    def get_query_ids(self) -> BoolTensor:
        return self.__dataset.query_ids

    def get_gallery_ids(self) -> BoolTensor:
        return self.__dataset.gallery_ids

    def visualize(self, item: int, color: TColor = BLACK) -> NDArray[np.uint8]:
        return self.__dataset.visualize(item=item, color=color)

    def visualize_with_player(self, item: int, color: TColor = BLACK) -> str:
        return self.__dataset.visualize_with_player(item, color)


__all__ = [
    "AudioBaseDataset",
    "AudioLabeledDataset",
    "AudioQueryGalleryDataset",
    "AudioQueryGalleryLabeledDataset",
]
