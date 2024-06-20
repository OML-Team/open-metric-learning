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


def update_dataset_extra_data(dataset, df, extra_data):
    extra_data = dict() or extra_data

    if CATEGORIES_COLUMN in df.columns:
        extra_data[CATEGORIES_COLUMN] = df[CATEGORIES_COLUMN].copy()

    if SEQUENCE_COLUMN in df.columns:
        extra_data[SEQUENCE_COLUMN] = df[SEQUENCE_COLUMN].copy()

    dataset.extra_data.update(extra_data)

    return dataset


def label_to_category(df: pd.DataFrame) -> Dict[Union[str, int], str]:
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

        self.__dataset = update_dataset_extra_data(dataset, df, extra_data)

        self.df = df
        self.labels_key = labels_key
        self.index_key = self.__dataset.index_key
        self.input_tensors_key = self.__dataset.input_tensors_key

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = self.__dataset[item]
        data[self.labels_key] = self.df.iloc[item][LABELS_COLUMN]
        return data

    def get_labels(self) -> np.ndarray:
        return np.array(self.df[LABELS_COLUMN])

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        return label_to_category(self.df)


class DFQueryGalleryDataset(IQueryGalleryDataset):
    def __init__(
        self,
        dataset: IBaseDataset,
        df: pd.DataFrame,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN))

        self.__dataset = update_dataset_extra_data(dataset, df, extra_data)

        self.df = df
        self._query_ids = BoolTensor(self.df[IS_QUERY_COLUMN]).nonzero().squeeze()
        self._gallery_ids = BoolTensor(self.df[IS_GALLERY_COLUMN]).nonzero().squeeze()

        self.index_key = self.__dataset.index_key
        self.input_tensors_key = self.__dataset.input_tensors_key

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.__dataset[item]

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

        self.__dataset = update_dataset_extra_data(dataset, df, extra_data)

        self.df = df
        self._query_ids = BoolTensor(self.df[IS_QUERY_COLUMN]).nonzero().squeeze()
        self._gallery_ids = BoolTensor(self.df[IS_GALLERY_COLUMN]).nonzero().squeeze()

        self.index_key = self.__dataset.index_key
        self.input_tensors_key = self.__dataset.input_tensors_key
        self.labels_key = labels_key

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.__dataset[item]

    def get_query_ids(self) -> LongTensor:
        return self._query_ids

    def get_gallery_ids(self) -> LongTensor:
        return self._gallery_ids

    def get_labels(self) -> np.ndarray:
        return np.array(self.df[LABELS_COLUMN])

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        return label_to_category(self.df)


__all__ = ["DFLabeledDataset", "DFQueryGalleryDataset", "DFQueryGalleryLabeledDataset"]
