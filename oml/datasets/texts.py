from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from torch import BoolTensor, LongTensor

from oml.const import (
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    LABELS_KEY,
    TEXTS_COLUMN,
    TColor,
)
from oml.interfaces.datasets import (
    ILabeledDataset,
    IQueryGalleryDataset,
    IQueryGalleryLabeledDataset,
    IVisualizableDataset,
)

TTransformersTokenizer = Any


def text_as_img(text: str, color: TColor) -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 50)
    font_scale = 1
    thickness = 2

    img = cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

    return img


class TextLabeledDataset(ILabeledDataset, IVisualizableDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TTransformersTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert all(x in df.columns for x in (TEXTS_COLUMN, LABELS_COLUMN))
        self.extra_data = extra_data or dict()

        self._df = df
        self._tokenizer = tokenizer
        self._max_length = max_length

        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.index_key = index_key

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        text = self._df.iloc[item][TEXTS_COLUMN]
        label = self._df.iloc[item][LABELS_COLUMN]

        inputs = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {self.input_tensors_key: inputs.input_ids.squeeze(0), self.labels_key: label, self.index_key: item}

    def get_labels(self) -> np.ndarray:
        return np.array(self._df[LABELS_COLUMN])

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        img = text_as_img(text=self._df.iloc[item][TEXTS_COLUMN], color=color)
        return img


class TextQueryGalleryDataset(IQueryGalleryDataset, IVisualizableDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TTransformersTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert all(x in df.columns for x in (TEXTS_COLUMN, IS_QUERY_COLUMN, IS_GALLERY_COLUMN))

        self._df = df
        self._tokenizer = tokenizer
        self._max_length = max_length

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key
        self.extra_data = extra_data or dict()

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        text = self._df.iloc[item][TEXTS_COLUMN]

        inputs = self._tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {self.input_tensors_key: inputs.input_ids.squeeze(0), self.index_key: item}

    def get_query_ids(self) -> LongTensor:
        return BoolTensor(self._df[IS_QUERY_COLUMN]).nonzero().squeeze()

    def get_gallery_ids(self) -> LongTensor:
        return BoolTensor(self._df[IS_GALLERY_COLUMN]).nonzero().squeeze()

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        img = text_as_img(text=self._df.iloc[item][TEXTS_COLUMN], color=color)
        return img


class TextQueryGalleryLabeledDataset(TextQueryGalleryDataset, IQueryGalleryLabeledDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TTransformersTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert all(x in df.columns for x in (TEXTS_COLUMN, IS_QUERY_COLUMN, IS_GALLERY_COLUMN, LABELS_COLUMN))

        self.labels_key = labels_key

        super().__init__(
            df=df,
            tokenizer=tokenizer,
            max_length=max_length,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
            extra_data=extra_data,
        )

    def get_labels(self) -> np.ndarray:
        return np.array(self._df[LABELS_COLUMN])

    def ___getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        data[self.labels_key] = self._df.iloc[item][LABELS_COLUMN]
        return data


__all__ = ["TextLabeledDataset", "TextQueryGalleryDataset", "TextQueryGalleryLabeledDataset"]
