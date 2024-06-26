from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as albu
import numpy as np
import pandas as pd
import torchvision
from torch import FloatTensor

from oml.const import (
    BLACK,
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    LABELS_COLUMN,
    LABELS_KEY,
    PATHS_COLUMN,
    SPLIT_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
    TBBoxes,
    TColor,
)
from oml.datasets.dataframe import (
    DFLabeledDataset,
    DFQueryGalleryDataset,
    DFQueryGalleryLabeledDataset,
)
from oml.interfaces.datasets import (
    IBaseDataset,
    ILabeledDataset,
    IQueryGalleryLabeledDataset,
    IVisualizableDataset,
)
from oml.registry.transforms import get_transforms
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.images.images import TImReader, draw_bbox, imread_pillow


def parse_bboxes(df: pd.DataFrame) -> Optional[TBBoxes]:
    n_existing_columns = sum([x in df for x in [X1_COLUMN, Y1_COLUMN, X2_COLUMN, Y2_COLUMN]])

    if n_existing_columns == 4:
        bboxes = []
        for _, row in df.iterrows():
            bbox = int(row[X1_COLUMN]), int(row[Y1_COLUMN]), int(row[X2_COLUMN]), int(row[Y2_COLUMN])
            bbox = None if any(coord is None for coord in bbox) else bbox
            bboxes.append(bbox)

    elif n_existing_columns == 0:
        bboxes = None

    else:
        raise ValueError(f"Found {n_existing_columns} bounding bboxes columns instead of 4. Check your dataframe.")

    return bboxes


class ImageBaseDataset(IBaseDataset, IVisualizableDataset):
    """
    The base class that handles image specific logic.

    """

    def __init__(
        self,
        paths: List[Path],
        dataset_root: Optional[Union[str, Path]] = None,
        bboxes: Optional[TBBoxes] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        transform: Optional[TTransforms] = None,
        f_imread: Optional[TImReader] = None,
        cache_size: Optional[int] = 0,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        index_key: str = INDEX_KEY,
    ):
        """

        Args:
            paths: Paths to images. Will be concatenated with ``dataset_root`` if provided.
            dataset_root: Path to the images' dir, set ``None`` if you provided the absolute paths in your dataframe
            bboxes: Bounding boxes of images. Some of the images may not have bounding bboxes.
            extra_data: Dictionary containing records of some additional information.
            transform: Augmentations for the images, set ``None`` to perform only normalisation and casting to tensor
            f_imread: Function to read the images, pass ``None`` to pick it automatically based on provided transforms
            cache_size: Size of the dataset's cache
            input_tensors_key: Key to put tensors into the batches
            index_key: Key to put samples' ids into the batches

        """
        assert (bboxes is None) or (len(paths) == len(bboxes))

        if extra_data is not None:
            assert all(
                len(record) == len(paths) for record in extra_data.values()
            ), "All the extra records need to have the size equal to the dataset's size"
            self.extra_data = extra_data
        else:
            self.extra_data = {}

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

        if dataset_root is not None:
            paths = list(map(lambda x: Path(dataset_root) / x, paths))

        self._paths = list(map(str, paths))
        self._bboxes = bboxes
        self._transform = transform if transform else get_transforms("norm_albu")
        self._f_imread = f_imread or get_im_reader_for_transforms(self._transform)

        if cache_size:
            self.read_bytes = lru_cache(maxsize=cache_size)(self._read_bytes)  # type: ignore
        else:
            self.read_bytes = self._read_bytes  # type: ignore

        available_transforms = (albu.Compose, torchvision.transforms.Compose)
        assert isinstance(self._transform, available_transforms), f"Transforms must one of: {available_transforms}"

    @staticmethod
    def _read_bytes(path: Union[Path, str]) -> bytes:
        with open(str(path), "rb") as fin:
            return fin.read()

    def __getitem__(self, item: int) -> Dict[str, Union[FloatTensor, int]]:
        img_bytes = self.read_bytes(self._paths[item])
        img = self._f_imread(img_bytes)

        im_h, im_w = img.shape[:2] if isinstance(img, np.ndarray) else img.size[::-1]

        if (self._bboxes is not None) and (self._bboxes[item] is not None):
            x1, y1, x2, y2 = self._bboxes[item]
        else:
            x1, y1, x2, y2 = 0, 0, im_w, im_h

        if isinstance(self._transform, albu.Compose):
            img = img[y1:y2, x1:x2, :]
            image_tensor = self._transform(image=img)["image"]
        else:
            # torchvision.transforms
            img = img.crop((x1, y1, x2, y2))
            image_tensor = self._transform(img)

        data = {
            self.input_tensors_key: image_tensor,
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
        img = np.array(imread_pillow(self.read_bytes(self._paths[item])))
        bbox = self._bboxes[item] if (self._bboxes is not None) else None
        image = draw_bbox(im=img, bbox=bbox, color=color)

        return image


class ImageLabeledDataset(DFLabeledDataset, IVisualizableDataset):
    """
    The dataset of images having their ground truth labels.

    """

    _dataset: ImageBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        extra_data: Optional[Dict[str, Any]] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        transform: Optional[albu.Compose] = None,
        f_imread: Optional[TImReader] = None,
        cache_size: Optional[int] = 0,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        index_key: str = INDEX_KEY,
    ):
        dataset = ImageBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            bboxes=parse_bboxes(df),
            extra_data=None,
            dataset_root=dataset_root,
            transform=transform,
            f_imread=f_imread,
            cache_size=cache_size,
            input_tensors_key=input_tensors_key,
            index_key=index_key,
        )
        super().__init__(dataset=dataset, df=df, extra_data=extra_data, labels_key=labels_key)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)


class ImageQueryGalleryLabeledDataset(DFQueryGalleryLabeledDataset, IVisualizableDataset):
    """
    The annotated dataset of images having `query`/`gallery` split.

    Note, that some datasets used as benchmarks in Metric Learning
    explicitly provide the splitting information (for example, ``DeepFashion InShop`` dataset), but some of them
    don't (for example, ``CARS196`` or ``CUB200``). The validation idea for the latter is to perform `1 vs rest`
    validation, where every query is evaluated versus the whole validation dataset (except for this exact query).

    So, if you want an item participate in validation as both: query and gallery, you should mark this item as
    ``is_query == True`` and ``is_gallery == True``, as it's done in the `CARS196` or `CUB200` dataset.

    """

    _dataset: ImageBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        extra_data: Optional[Dict[str, Any]] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        transform: Optional[albu.Compose] = None,
        f_imread: Optional[TImReader] = None,
        cache_size: Optional[int] = 0,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
    ):
        dataset = ImageBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            bboxes=parse_bboxes(df),
            extra_data=None,
            dataset_root=dataset_root,
            transform=transform,
            f_imread=f_imread,
            cache_size=cache_size,
            input_tensors_key=input_tensors_key,
        )

        super().__init__(dataset=dataset, df=df, extra_data=extra_data, labels_key=labels_key)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)


class ImageQueryGalleryDataset(DFQueryGalleryDataset, IVisualizableDataset):
    """
    The NOT annotated dataset of images having `query`/`gallery` split.

    """

    _dataset: ImageBaseDataset

    def __init__(
        self,
        df: pd.DataFrame,
        extra_data: Optional[Dict[str, Any]] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        transform: Optional[albu.Compose] = None,
        f_imread: Optional[TImReader] = None,
        cache_size: Optional[int] = 0,
        input_tensors_key: str = INPUT_TENSORS_KEY,
    ):
        dataset = ImageBaseDataset(
            paths=df[PATHS_COLUMN].tolist(),
            bboxes=parse_bboxes(df),
            extra_data=None,
            dataset_root=dataset_root,
            transform=transform,
            f_imread=f_imread,
            cache_size=cache_size,
            input_tensors_key=input_tensors_key,
        )
        super().__init__(dataset=dataset, df=df, extra_data=extra_data)

    def visualize(self, item: int, color: TColor) -> np.ndarray:
        return self._dataset.visualize(item=item, color=color)


def get_retrieval_images_datasets(
    dataset_root: Path,
    transforms_train: Any,
    transforms_val: Any,
    f_imread_train: Optional[TImReader] = None,
    f_imread_val: Optional[TImReader] = None,
    dataframe_name: str = "df.csv",
    cache_size: Optional[int] = 0,
    verbose: bool = True,
) -> Tuple[ILabeledDataset, IQueryGalleryLabeledDataset]:
    df = pd.read_csv(dataset_root / dataframe_name, index_col=False)

    check_retrieval_dataframe_format(df, dataset_root=dataset_root, verbose=verbose)

    # first half will consist of "train" split, second one of "val"
    # so labels in train will be from 0 to N-1 and labels in test will be from N to K
    mapper = {l: i for i, l in enumerate(df.sort_values(by=[SPLIT_COLUMN])[LABELS_COLUMN].unique())}

    # train
    df_train = df[df[SPLIT_COLUMN] == "train"].reset_index(drop=True)
    df_train[LABELS_COLUMN] = df_train[LABELS_COLUMN].map(mapper)

    train_dataset = ImageLabeledDataset(
        df=df_train,
        dataset_root=dataset_root,
        transform=transforms_train,
        cache_size=cache_size,
        f_imread=f_imread_train,
    )

    # val (query + gallery)
    df_query_gallery = df[df[SPLIT_COLUMN] == "validation"].reset_index(drop=True)
    valid_dataset = ImageQueryGalleryLabeledDataset(
        df=df_query_gallery,
        dataset_root=dataset_root,
        transform=transforms_val,
        cache_size=cache_size,
        f_imread=f_imread_val,
    )

    return train_dataset, valid_dataset


__all__ = [
    "ImageBaseDataset",
    "ImageLabeledDataset",
    "ImageQueryGalleryDataset",
    "ImageQueryGalleryLabeledDataset",
    "get_retrieval_images_datasets",
]
