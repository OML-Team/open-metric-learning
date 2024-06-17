import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from numpy.typing import NDArray
from torch import BoolTensor, FloatTensor, LongTensor
from torchaudio.transforms import MelSpectrogram, Resample

from oml.const import (
    AUDIO_EXTENSIONS,
    BLACK,
    CATEGORIES_COLUMN,
    DEFAULT_DURATION,
    DEFAULT_MELSPEC_PARAMS,
    DEFAULT_NUM_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    INDEX_KEY,
    INPUT_FRAME_OFFSET_KEY,
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
    IVisualizableDataset,
)
from oml.utils.audios import visualize_audio, visualize_audio_html


class AudioBaseDataset(IBaseDataset, IVisualizableDataset):
    """
    The base class that handles audio specific logic.

    """

    def __init__(
        self,
        paths: List[Union[str, Path]],
        dataset_root: Optional[Union[str, Path]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        sr: int = DEFAULT_SAMPLE_RATE,
        num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        input_frame_offset_key: str = INPUT_FRAME_OFFSET_KEY,
        index_key: str = INDEX_KEY,
    ):
        """
        Initializes the AudioDataset.

        Args:
            paths (List[Union[str, Path]]): List of audio file paths.
            dataset_root (Optional[Union[str, Path]]): Base path for audio files, optional.
            extra_data (Optional[Dict[str, Any]]): Extra data to include in dataset items.
            sr (int): Sampling rate of audio files.
            num_seconds (float): Duration to use from each audio file.
            num_channels (int): Number of audio channels.
            input_tensors_key (str): Key under which audio tensors are stored.
            input_frame_offset_key (str): Key under which audio offsets are stored
            index_key (str): Key for indexing dataset items.
        """
        self.sr = sr
        self.num_frames = int(num_seconds * sr)
        self.num_channels = num_channels

        if extra_data is not None:
            assert all(
                len(record) == len(paths) for record in extra_data.values()
            ), "All the extra records need to have the size equal to the dataset's size"
            self.extra_data = extra_data
        else:
            self.extra_data = {}

        self.frame_offsets: List[int] = self.extra_data.get(input_frame_offset_key)

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

        self._paths = [Path(p) if dataset_root is None else Path(dataset_root) / p for p in paths]

    def _downmix_and_resample(self, audio: FloatTensor, sr: int) -> FloatTensor:
        """
        Downmix audio to mono and resample it to the dataset's sampling rate.

        Args:
            audio (FloatTensor): Input audio tensor.
            sr (int): Original sampling rate of the audio.

        Returns:
            FloatTensor: Processed audio tensor.
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
            audio (FloatTensor): Audio tensor.
            frame_offset (int): Starting frame offset for trimming.

        Returns:
            FloatTensor: Trimmed or padded audio tensor.
        """
        audio_length = audio.shape[1]
        if audio_length > self.num_frames:
            if frame_offset is None:
                frame_offset = random.randrange(0, audio_length - self.num_frames)
            audio = audio[:, frame_offset : frame_offset + self.num_frames]
        else:
            padding = (self.num_frames - audio_length, 0)
            audio = torch.nn.functional.pad(audio, padding)
        return audio

    def get_audio(self, item: int) -> FloatTensor:
        """
        Load and process an audio file.

        Args:
            item (int): Dataset item index.

        Returns:
            FloatTensor: Processed audio tensor.
        """
        path = self._paths[item]
        assert (
            path.suffix in AUDIO_EXTENSIONS
        ), f"Input audio has invalid extension '{path.suffix}'. It should be one of {AUDIO_EXTENSIONS}."
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
            audio (FloatTensor): The input audio tensor.
            params (Dict[str, Any]): Parameters for spectral representation computation.

        Returns:
            FloatTensor: The spectral representation of the input audio tensor.
        """
        melspectrogram = MelSpectrogram(sample_rate=self.sr, **params)
        melspec = melspectrogram(audio)
        log_melspec = torch.log1p(melspec)
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
            item (int): Dataset item index.
            color (str): Color of the plot.

        Returns:
            NDArray: Array representing the image of the plot.
        """
        audio = self.get_audio(item)
        spec_repr = self.get_spectral_repr(audio).squeeze(0)
        return visualize_audio(spec_repr=spec_repr, color=color)

    def visualize_html(self, item: int, color: str = "black") -> str:
        """
        Visualize an audio file in HTML markup.

        Args:
            item (int): Dataset item index.
            color (str): Color of the plot.

        Returns:
            str: HTML markup with spectral representation image and audio player.
        """
        audio = self.get_audio(item)
        spec_repr = self.get_spectral_repr(audio).squeeze(0)
        return visualize_audio_html(audio=audio, spec_repr=spec_repr, sr=self.sr, color=color)


class AudioLabeledDataset(AudioBaseDataset, ILabeledDataset):
    """
    The dataset of audios having their ground truth labels.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Optional[Union[str, Path]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        sr: int = DEFAULT_SAMPLE_RATE,
        num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        input_frame_offset_key: str = INPUT_FRAME_OFFSET_KEY,
        index_key: str = INDEX_KEY,
        labels_key: str = LABELS_KEY,
    ):
        assert all(x in df.columns for x in (LABELS_COLUMN, PATHS_COLUMN))
        self.labels_key = labels_key
        self.df = df

        extra_data = extra_data or dict()

        if CATEGORIES_COLUMN in df.columns:
            extra_data[CATEGORIES_COLUMN] = df[CATEGORIES_COLUMN].copy()

        if SEQUENCE_COLUMN in df.columns:
            extra_data[SEQUENCE_COLUMN] = df[SEQUENCE_COLUMN].copy()

        super().__init__(
            paths=df[PATHS_COLUMN],
            dataset_root=dataset_root,
            extra_data=extra_data,
            sr=sr,
            num_seconds=num_seconds,
            num_channels=num_channels,
            input_tensors_key=input_tensors_key,
            input_frame_offset_key=input_frame_offset_key,
            index_key=index_key,
        )

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        data[self.labels_key] = self.df.iloc[item][LABELS_COLUMN]
        return data

    def get_labels(self) -> NDArray[np.int64]:
        return np.array(self.df[LABELS_COLUMN])

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        if CATEGORIES_COLUMN in self.df.columns:
            label2category = dict(zip(self.df[LABELS_COLUMN], self.df[CATEGORIES_COLUMN]))
        else:
            label2category = None

        return label2category


class AudioQueryGalleryDataset(AudioBaseDataset, IQueryGalleryDataset):
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
        sr: int = DEFAULT_SAMPLE_RATE,
        num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        input_frame_offset_key: str = INPUT_FRAME_OFFSET_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN, PATHS_COLUMN))
        self.df = df

        super().__init__(
            paths=df[PATHS_COLUMN],
            dataset_root=dataset_root,
            extra_data=extra_data,
            sr=sr,
            num_seconds=num_seconds,
            num_channels=num_channels,
            input_tensors_key=input_tensors_key,
            input_frame_offset_key=input_frame_offset_key,
            index_key=index_key,
        )

    def get_query_ids(self) -> LongTensor:
        return BoolTensor(self.df[IS_QUERY_COLUMN]).nonzero().squeeze()

    def get_gallery_ids(self) -> LongTensor:
        return BoolTensor(self.df[IS_GALLERY_COLUMN]).nonzero().squeeze()


class AudioQueryGalleryLabeledDataset(AudioQueryGalleryDataset, AudioLabeledDataset):
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
        sr: int = DEFAULT_SAMPLE_RATE,
        num_seconds: float = DEFAULT_DURATION,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        input_frame_offset_key: str = INPUT_FRAME_OFFSET_KEY,
        index_key: str = INDEX_KEY,
        labels_key: str = LABELS_KEY,
    ):
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN, LABELS_COLUMN, PATHS_COLUMN))

        AudioLabeledDataset.__init__(
            self,
            df=df,
            dataset_root=dataset_root,
            extra_data=extra_data,
            sr=sr,
            num_seconds=num_seconds,
            num_channels=num_channels,
            input_tensors_key=input_tensors_key,
            input_frame_offset_key=input_frame_offset_key,
            index_key=index_key,
            labels_key=labels_key,
        )
