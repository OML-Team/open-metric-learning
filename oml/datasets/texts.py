from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from torch import BoolTensor, LongTensor

from oml.const import (
    BLACK,
    CATEGORIES_COLUMN,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    LABELS_KEY,
    SEQUENCE_COLUMN,
    TEXTS_COLUMN,
    TColor,
)
from oml.interfaces.datasets import (
    IBaseDataset,
    ILabeledDataset,
    IQueryGalleryDataset,
    IQueryGalleryLabeledDataset,
    IVisualizableDataset,
)
from oml.utils.misc import visualise_text

THuggingFaceTokenizer = Any


class TextBaseDataset(IBaseDataset, IVisualizableDataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: THuggingFaceTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
    ):
        if extra_data is not None:
            assert all(
                len(record) == len(texts) for record in extra_data.values()
            ), "All the extra records need to have the size equal to the dataset's size"
            self.extra_data = extra_data
        else:
            self.extra_data = {}

        self._texts = texts
        self._tokenizer = tokenizer
        self._max_length = max_length

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        encoded_text = self._tokenizer(
            self._texts[item],
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        encoded_text["input_ids"] = encoded_text["input_ids"].squeeze()
        encoded_text["attention_mask"] = encoded_text["attention_mask"].squeeze()

        data = {self.input_tensors_key: encoded_text, self.index_key: item}

        for key, record in self.extra_data.items():
            if key in data:
                raise ValueError(f"<extra_data> and dataset share the same key: {key}")
            else:
                data[key] = record[item]

        return data

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        img = visualise_text(text=self._texts[item], color=color)
        return img


class TextLabeledDataset(TextBaseDataset, ILabeledDataset):
    """
    The dataset of texts having their ground truth labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: THuggingFaceTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert all(x in df.columns for x in (TEXTS_COLUMN, LABELS_COLUMN))
        self.labels_key = labels_key
        self.df = df

        extra_data = extra_data or dict()

        if CATEGORIES_COLUMN in df.columns:
            extra_data[CATEGORIES_COLUMN] = df[CATEGORIES_COLUMN].copy()

        if SEQUENCE_COLUMN in df.columns:
            extra_data[SEQUENCE_COLUMN] = df[SEQUENCE_COLUMN].copy()

        super().__init__(
            texts=df[TEXTS_COLUMN],
            tokenizer=tokenizer,
            max_length=max_length,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
        )

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        data[self.labels_key] = self.df.iloc[item][LABELS_COLUMN]
        return data

    def get_labels(self) -> np.ndarray:
        return np.array(self.df[LABELS_COLUMN])

    def get_label2category(self) -> Optional[Dict[int, Union[str, int]]]:
        if CATEGORIES_COLUMN in self.df.columns:
            label2category = dict(zip(self.df[LABELS_COLUMN], self.df[CATEGORIES_COLUMN]))
        else:
            label2category = None

        return label2category


class TextQueryGalleryLabeledDataset(TextLabeledDataset, IQueryGalleryLabeledDataset):
    """
    The annotated dataset of texts having `query`/`gallery` split.
    To perform `1 vs rest` validation, where a query is evaluated versus the whole validation dataset
    (except for this exact query), you should mark the item as ``is_query == True`` and ``is_gallery == True``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: THuggingFaceTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        labels_key: str = LABELS_KEY,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert all(x in df.columns for x in (TEXTS_COLUMN, IS_QUERY_COLUMN, IS_GALLERY_COLUMN, LABELS_COLUMN))

        self.df = df
        super().__init__(
            df=df,
            tokenizer=tokenizer,
            max_length=max_length,
            extra_data=extra_data,
            input_tensors_key=input_tensors_key,
            labels_key=labels_key,
            index_key=index_key,
        )

    def get_query_ids(self) -> LongTensor:
        return BoolTensor(self.df[IS_QUERY_COLUMN]).nonzero().squeeze()

    def get_gallery_ids(self) -> LongTensor:
        return BoolTensor(self.df[IS_GALLERY_COLUMN]).nonzero().squeeze()


class TextQueryGalleryDataset(IVisualizableDataset, IQueryGalleryDataset):
    """
    The NOT annotated dataset of texts having `query`/`gallery` split.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: THuggingFaceTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert all(x in df.columns for x in (TEXTS_COLUMN, IS_QUERY_COLUMN, IS_GALLERY_COLUMN))

        # instead of implementing the whole logic let's just re-use QGL dataset, but with dropped labels
        df = deepcopy(df)
        df[LABELS_COLUMN] = "fake_label"

        self.__dataset = TextQueryGalleryLabeledDataset(
            df=df,
            extra_data=extra_data,
            max_length=max_length,
            tokenizer=tokenizer,
            input_tensors_key=input_tensors_key,
            labels_key=LABELS_COLUMN,
            index_key=index_key,
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

    def get_query_ids(self) -> LongTensor:
        return self.__dataset.get_query_ids()

    def get_gallery_ids(self) -> LongTensor:
        return self.__dataset.get_gallery_ids()

    def visualize(self, item: int, color: TColor = BLACK) -> np.ndarray:
        return self.__dataset.visualize(item=item, color=color)


__all__ = ["TextBaseDataset", "TextLabeledDataset", "TextQueryGalleryDataset", "TextQueryGalleryLabeledDataset"]
