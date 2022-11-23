from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import albumentations as albu
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset

from oml.const import (
    CATEGORIES_COLUMN,
    CATEGORIES_KEY,
    INPUT_TENSORS_KEY,
    IS_GALLERY_COLUMN,
    IS_GALLERY_KEY,
    IS_QUERY_COLUMN,
    IS_QUERY_KEY,
    LABELS_COLUMN,
    LABELS_KEY,
    PATHS_COLUMN,
    PATHS_KEY,
    SPLIT_COLUMN,
    X1_COLUMN,
    X1_KEY,
    X2_COLUMN,
    X2_KEY,
    Y1_COLUMN,
    Y1_KEY,
    Y2_COLUMN,
    Y2_KEY,
)
from oml.interfaces.datasets import IDatasetQueryGallery, IDatasetWithLabels
from oml.registry.transforms import get_transforms
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.images.images import TImReader, imread_cv2


class BaseDataset(Dataset):
    """
    Base class for the retrieval datasets.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[TTransforms] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        f_imread: TImReader = imread_cv2,
        cache_size: int = 100_000,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        paths_key: str = PATHS_KEY,
        categories_key: Optional[str] = CATEGORIES_KEY,
        x1_key: str = X1_KEY,
        x2_key: str = X2_KEY,
        y1_key: str = Y1_KEY,
        y2_key: str = Y2_KEY,
    ):
        """

        Args:
            df: Table with the following obligatory columns:

                  >>> LABELS_COLUMN, PATHS_COLUMN

                  and the optional ones:

                  >>> X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN, CATEGORIES_COLUMN

            transform: Augmentations for the images, set ``None`` to perform only normalisation and casting to tensor
            dataset_root: Path to the images dir, set ``None`` if you provided the absolute paths in your dataframe
            f_imread: Function to read the images
            cache_size: Size of the dataset's cache
            input_tensors_key: Key to put tensors into the batches
            labels_key: Key to put labels into the batches
            paths_key: Key put paths into the batches
            categories_key: Key to put categories into the batches
            x1_key: Key to put ``x1`` into the batches
            x2_key: Key to put ``x2`` into the batches
            y1_key: Key to put ``y1`` into the batches
            y2_key: Key to put ``y2`` into the batches

        """
        df = df.copy()

        assert all(x in df.columns for x in (LABELS_COLUMN, PATHS_COLUMN))

        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.paths_key = paths_key
        self.categories_key = categories_key if (CATEGORIES_COLUMN in df.columns) else None

        self.bboxes_exist = all(coord in df.columns for coord in (X1_COLUMN, X2_COLUMN, Y1_COLUMN, Y2_COLUMN))
        if self.bboxes_exist:
            self.x1_key, self.x2_key, self.y1_key, self.y2_key = x1_key, x2_key, y1_key, y2_key
        else:
            self.x1_key, self.x2_key, self.y1_key, self.y2_key = None, None, None, None

        if dataset_root is not None:
            dataset_root = Path(dataset_root)
            df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: str(dataset_root / x))
        else:
            df[PATHS_COLUMN] = df[PATHS_COLUMN].astype(str)

        self.df = df
        self.transform = transform if transform else get_transforms("norm_albu")
        self.f_imread = f_imread
        self.read_bytes_image_cached = lru_cache(maxsize=cache_size)(self._read_bytes_image)

        available_augs_types = (albu.Compose, torchvision.transforms.Compose)
        assert isinstance(self.transform, available_augs_types), f"Type of transforms must be in {available_augs_types}"

    @staticmethod
    def _read_bytes_image(path: Union[Path, str]) -> bytes:
        with open(str(path), "rb") as fin:
            return fin.read()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_bytes = self.read_bytes_image_cached(row[PATHS_COLUMN])
        img = self.f_imread(img_bytes)

        im_h, im_w = img.shape[:2] if isinstance(img, np.ndarray) else img.size[::-1]

        if (not self.bboxes_exist) or any(
            pd.isna(coord) for coord in [row[X1_COLUMN], row[X2_COLUMN], row[Y1_COLUMN], row[Y2_COLUMN]]
        ):
            x1, y1, x2, y2 = 0, 0, im_w, im_h
        else:
            x1, y1, x2, y2 = int(row[X1_COLUMN]), int(row[Y1_COLUMN]), int(row[X2_COLUMN]), int(row[Y2_COLUMN])

        if isinstance(self.transform, albu.Compose):
            img = img[y1:y2, x1:x2, :]  # todo: since albu may handle bboxes we should move it to augs
            image_tensor = self.transform(image=img)["image"]
        else:
            # torchvision.transforms
            img = img.crop((x1, y1, x2, y2))
            image_tensor = self.transform(img)

        item = {
            self.input_tensors_key: image_tensor,
            self.labels_key: row[LABELS_COLUMN],
            self.paths_key: row[PATHS_COLUMN],
        }

        if self.categories_key:
            item[self.categories_key] = row[CATEGORIES_COLUMN]

        if self.bboxes_exist:
            item.update(
                {
                    self.x1_key: x1,
                    self.y1_key: y1,
                    self.x2_key: x2,
                    self.y2_key: y2,
                }
            )

        return item

    def __len__(self) -> int:
        return len(self.df)

    @property
    def bboxes_keys(self) -> Tuple[str, ...]:
        if self.bboxes_exist:
            return self.x1_key, self.y1_key, self.x2_key, self.y2_key
        else:
            return tuple()


class DatasetWithLabels(BaseDataset, IDatasetWithLabels):
    """
    The main purpose of this class is to be used as a dataset during
    the training stage.

    It has to know how to return its labels, which is required information
    to perform the training with the combinations-based losses.
    Particularly, these labels will be passed to `Sampler` to form the batches and
    batches will be passed to `Miner` to form the combinations (triplets).

    """

    def get_labels(self) -> np.ndarray:
        return np.array(self.df[LABELS_COLUMN].tolist())


class DatasetQueryGallery(BaseDataset, IDatasetQueryGallery):
    """
    The main purpose of this class is to be used as a dataset during
    the validation stage. It has to provide information
    about its `query`/`gallery` split.

    Note, that some of the datasets used as benchmarks in Metric Learning
    provide the splitting information (for example, ``DeepFashion InShop`` dataset), but some of them
    don't (for example, ``CARS196`` or ``CUB200``).
    The validation idea for the latter is to calculate the embeddings for the whole validation set,
    then for every item find ``top-k`` nearest neighbors and calculate the desired retrieval metric.
    In other words, for the desired query item, the gallery is the rest of the validation dataset.

    Thus, if you want to perform this kind of validation process (`1 vs rest`) you should simply return
    ``is_query == True`` and ``is_gallery == True`` for every item in the dataset sa the same time.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Optional[Union[str, Path]] = None,
        transform: Optional[albu.Compose] = None,
        f_imread: TImReader = imread_cv2,
        cache_size: int = 100_000,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        paths_key: str = PATHS_KEY,
        categories_key: str = CATEGORIES_KEY,
        x1_key: str = X1_KEY,
        x2_key: str = X2_KEY,
        y1_key: str = Y1_KEY,
        y2_key: str = Y2_KEY,
        is_query_key: str = IS_QUERY_KEY,
        is_gallery_key: str = IS_GALLERY_KEY,
    ):
        super(DatasetQueryGallery, self).__init__(
            df=df,
            dataset_root=dataset_root,
            transform=transform,
            f_imread=f_imread,
            cache_size=cache_size,
            input_tensors_key=input_tensors_key,
            labels_key=labels_key,
            paths_key=paths_key,
            categories_key=categories_key,
            x1_key=x1_key,
            x2_key=x2_key,
            y1_key=y1_key,
            y2_key=y2_key,
        )
        assert all(x in df.columns for x in (IS_QUERY_COLUMN, IS_GALLERY_COLUMN))

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item[self.is_query_key] = bool(self.df.iloc[idx][IS_QUERY_COLUMN])
        item[self.is_gallery_key] = bool(self.df.iloc[idx][IS_GALLERY_COLUMN])
        return item


def get_retrieval_datasets(
    dataset_root: Path,
    transforms_train: Any,
    transforms_val: Any,
    f_imread_train: Optional[TImReader] = None,
    f_imread_val: Optional[TImReader] = None,
    dataframe_name: str = "df.csv",
    cache_size: int = 100_000,
    verbose: bool = True,
) -> Tuple[DatasetWithLabels, DatasetQueryGallery]:
    df = pd.read_csv(dataset_root / dataframe_name, index_col=False)

    check_retrieval_dataframe_format(df, dataset_root=dataset_root, verbose=verbose)

    # first half will consist of "train" split, second one of "val"
    # so labels in train will be from 0 to N-1 and labels in test will be from N to K
    mapper = {l: i for i, l in enumerate(df.sort_values(by=[SPLIT_COLUMN])[LABELS_COLUMN].unique())}

    # train
    df_train = df[df[SPLIT_COLUMN] == "train"].reset_index(drop=True)
    df_train[LABELS_COLUMN] = df_train[LABELS_COLUMN].map(mapper)

    train_dataset = DatasetWithLabels(
        df=df_train,
        dataset_root=dataset_root,
        transform=transforms_train,
        cache_size=cache_size,
        f_imread=f_imread_train or get_im_reader_for_transforms(transforms_train),
    )

    # val (query + gallery)
    df_query_gallery = df[df[SPLIT_COLUMN] == "validation"].reset_index(drop=True)
    valid_dataset = DatasetQueryGallery(
        df=df_query_gallery,
        dataset_root=dataset_root,
        transform=transforms_val,
        cache_size=cache_size,
        f_imread=f_imread_val or get_im_reader_for_transforms(transforms_val),
    )

    return train_dataset, valid_dataset


__all__ = ["BaseDataset", "DatasetWithLabels", "DatasetQueryGallery", "get_retrieval_datasets"]
