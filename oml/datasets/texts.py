import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from oml.const import INDEX_KEY, INPUT_TENSORS_KEY, LABELS_KEY, TEXTS_COLUMN, TColor
from oml.datasets.dataframe import (
    DFLabeledDataset,
    DFQueryGalleryDataset,
    DFQueryGalleryLabeledDataset,
)
from oml.interfaces.datasets import IBaseDataset, IVisualizableDataset
from oml.utils.misc import visualise_text

TTokenizer = Any


def check_tokenizer_type(tokenizer):  # type: ignore
    try:
        import transformers as t

        if not isinstance(tokenizer, (t.PreTrainedTokenizer, t.PreTrainedTokenizerFast)):
            warnings.warn(f"Unexpected tokenizer type: {type(tokenizer)}.")
    except ImportError:
        pass


class TextBaseDataset(IBaseDataset, IVisualizableDataset):
    """
    The base class that handles text specific logic.

    """

    def __init__(
        self,
        texts: List[str],
        tokenizer: TTokenizer,
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

        check_tokenizer_type(tokenizer)

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


class TextLabeledDataset(DFLabeledDataset, IVisualizableDataset):
    """
    The dataset of texts having their ground truth labels.

    """

    _dataset: TextBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        index_key: str = INDEX_KEY,
    ):
        dataset = TextBaseDataset(
            texts=df[TEXTS_COLUMN],
            tokenizer=tokenizer,
            max_length=max_length,
            extra_data=None,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
        )

        super().__init__(dataset=dataset, df=df, extra_data=extra_data, labels_key=labels_key)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)


class TextQueryGalleryLabeledDataset(DFQueryGalleryLabeledDataset, IVisualizableDataset):
    """
    The annotated dataset of texts having `query`/`gallery` split.
    To perform `1 vs rest` validation, where a query is evaluated versus the whole validation dataset
    (except for this exact query), you should mark the item as ``is_query == True`` and ``is_gallery == True``.

    """

    _dataset: TextBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        labels_key: str = LABELS_KEY,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
    ):
        dataset = TextBaseDataset(
            texts=df[TEXTS_COLUMN],
            tokenizer=tokenizer,
            max_length=max_length,
            extra_data=None,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
        )
        super().__init__(dataset=dataset, df=df, extra_data=extra_data, labels_key=labels_key)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)


class TextQueryGalleryDataset(DFQueryGalleryDataset, IVisualizableDataset):
    """
    The NOT annotated dataset of texts having `query`/`gallery` split.

    """

    _dataset: TextBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TTokenizer,
        max_length: int = 128,
        extra_data: Optional[Dict[str, Any]] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
    ):
        dataset = TextBaseDataset(
            texts=df[TEXTS_COLUMN],
            tokenizer=tokenizer,
            max_length=max_length,
            extra_data=None,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
        )

        super().__init__(dataset=dataset, df=df, extra_data=extra_data)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)


__all__ = ["TextBaseDataset", "TextLabeledDataset", "TextQueryGalleryDataset", "TextQueryGalleryLabeledDataset"]
