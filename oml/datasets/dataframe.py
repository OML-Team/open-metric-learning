from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from torch import BoolTensor, LongTensor

from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    LABELS_KEY,
    SEQUENCE_COLUMN,
)
from oml.interfaces.datasets import (
    IBaseDataset,
    ILabeledDataset,
    IQueryGalleryDataset,
    IQueryGalleryLabeledDataset,
)


def update_extra_data(dataset: IBaseDataset, df: pd.DataFrame, extra_data: Dict[str, Any]) -> IBaseDataset:
    extra_data = dict() if extra_data is None else extra_data

    if CATEGORIES_COLUMN in df.columns:
        extra_data[CATEGORIES_COLUMN] = df[CATEGORIES_COLUMN].copy()

    if SEQUENCE_COLUMN in df.columns:
        extra_data[SEQUENCE_COLUMN] = df[SEQUENCE_COLUMN].copy()

    dataset.extra_data.update(extra_data)

    return dataset


def build_label2category(df: pd.DataFrame) -> Optional[Dict[int, Union[str, int]]]:
    if CATEGORIES_COLUMN in df.columns:
        label2category = dict(zip(df[LABELS_COLUMN], df[CATEGORIES_COLUMN]))
    else:
        label2category = None

    return label2category


class DFLabeledDataset(ILabeledDataset):
    def __init__(
        self,
        dataset: IBaseDataset,
        df: pd.DataFrame,
        extra_data: Optional[Dict[str, Any]] = None,
        labels_key: str = LABELS_KEY,
    ):
        assert LABELS_COLUMN in df.columns
        dataset = update_extra_data(dataset, df, extra_data)

        self._dataset = dataset
        self._labels = np.array(df[LABELS_COLUMN].copy())
        self._label2category = build_label2category(df)

        self.df = df
        self.extra_data = self._dataset.extra_data
        self.labels_key = labels_key
        self.index_key = self._dataset.index_key
        self.input_tensors_key = self._dataset.input_tensors_key

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = self._dataset[item]
        data[self.labels_key] = self._labels[item]
        return data

    def get_labels(self) -> np.ndarray:
        return self._labels

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        return self._label2category


class DFQueryGalleryDataset(IQueryGalleryDataset):
    def __init__(
        self,
        dataset: IBaseDataset,
        df: pd.DataFrame,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN))
        dataset = update_extra_data(dataset, df, extra_data)

        self._dataset = dataset
        self._query_ids = BoolTensor(df[IS_QUERY_COLUMN]).nonzero().squeeze()
        self._gallery_ids = BoolTensor(df[IS_GALLERY_COLUMN]).nonzero().squeeze()

        self.df = df
        self.extra_data = self._dataset.extra_data
        self.index_key = self._dataset.index_key
        self.input_tensors_key = self._dataset.input_tensors_key

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self._dataset[item]

    def get_query_ids(self) -> LongTensor:
        return self._query_ids

    def get_gallery_ids(self) -> LongTensor:
        return self._gallery_ids


class DFQueryGalleryLabeledDataset(IQueryGalleryLabeledDataset):
    def __init__(
        self,
        dataset: IBaseDataset,
        df: pd.DataFrame,
        extra_data: Optional[Dict[str, Any]] = None,
        labels_key: str = LABELS_KEY,
    ):
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN, LABELS_COLUMN))
        dataset = update_extra_data(dataset, df, extra_data)

        self._dataset = dataset
        self._query_ids = BoolTensor(df[IS_QUERY_COLUMN]).nonzero().squeeze()
        self._gallery_ids = BoolTensor(df[IS_GALLERY_COLUMN]).nonzero().squeeze()
        self._labels = np.array(df[LABELS_COLUMN].copy())
        self._label2category = build_label2category(df)

        self.df = df
        self.extra_data = self._dataset.extra_data
        self.index_key = self._dataset.index_key
        self.input_tensors_key = self._dataset.input_tensors_key
        self.labels_key = labels_key

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = self._dataset[item]
        data[self.labels_key] = self._labels[item]
        return data

    def get_query_ids(self) -> LongTensor:
        return self._query_ids

    def get_gallery_ids(self) -> LongTensor:
        return self._gallery_ids

    def get_labels(self) -> np.ndarray:
        return self._labels

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        return self._label2category


__all__ = ["DFLabeledDataset", "DFQueryGalleryDataset", "DFQueryGalleryLabeledDataset"]
