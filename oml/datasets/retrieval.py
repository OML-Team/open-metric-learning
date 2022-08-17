from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import albumentations as albu
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from PIL.Image import Image as TImage
from torch.utils.data import Dataset

from oml.interfaces.datasets import IDatasetQueryGallery, IDatasetWithLabels
from oml.registry.transforms import TAugs
from oml.transforms.images.albumentations.shared import get_normalisation_albu
from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.images.images import TImReader, imread_cv2
from oml.utils.images.images_resize import pad_resize


class BaseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        im_size: int,
        pad_ratio: float = 0.0,
        dataset_root: Optional[Union[str, Path]] = None,
        transform: Optional[TAugs] = None,
        f_imread: TImReader = imread_cv2,
        cache_size: int = 100_000,
        input_tensors_key: str = "input_tensors",
        labels_key: str = "labels",
        paths_key: str = "paths",
        x1_key: str = "x1",
        x2_key: str = "x2",
        y1_key: str = "y1",
        y2_key: str = "y2",
    ):
        """

        Args:
            df: Table with the following columns:
                  obligatory: "label" - id of the item,
                              "path" - to the image, absolute or relative from "dataset_root"
                  optional: "x_1", "x_2", "y_1", "y_2" (left, right, top, bot)
            im_size: Images will be resized to (image_size, image_size)
            pad_ratio: Padding ratio
            dataset_root: Path to the images dir, set None if you provided the absolute paths
            transform: Augmentations for the images, set None to skip it
            f_imread: Function to read the image
            input_tensors_key: Key to get input_tensors from batch
            labels_key: Key to get labels from batch
            paths_key: Key to get paths from batch
            x1_key: Key to get x1 from batch
            x2_key: Key to get x2 from batch
            y1_key: Key to get y1 from batch
            y2_key: Key to get y2 from batch

        """
        assert pad_ratio >= 0
        assert all(x in df.columns for x in ("label", "path"))

        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.paths_key = paths_key
        self.x1_key, self.x2_key, self.y1_key, self.y2_key = x1_key, x2_key, y1_key, y2_key

        if not all(coord in df.columns for coord in ("x_1", "x_2", "y_1", "y_2")):
            df["x_1"] = None
            df["x_2"] = None
            df["y_1"] = None
            df["y_2"] = None

        if dataset_root is not None:
            dataset_root = Path(dataset_root)
            df["path"] = df["path"].apply(lambda x: str(dataset_root / x))
        else:
            df["path"] = df["path"].astype(str)

        self.df = df
        self.im_size = im_size
        self.pad_ratio = pad_ratio
        self.transform = transform or get_normalisation_albu()
        self.f_imread = f_imread
        self.read_bytes_image_cached = lru_cache(maxsize=cache_size)(self._read_bytes_image)

    @staticmethod
    def _read_bytes_image(path: Union[Path, str]) -> bytes:
        with open(str(path), "rb") as fin:
            return fin.read()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        label = self.df.iloc[idx]["label"]
        crop = self.read_image(idx)

        if isinstance(self.transform, albu.Compose):
            image_tensor = self.transform(image=crop)["image"]

        elif isinstance(self.transform, torchvision.transforms.Compose):
            if isinstance(crop, np.ndarray):  # depends on the reader we may have numpy or pil here
                crop = Image.fromarray(crop)
            image_tensor = self.transform(crop)

        else:
            image_tensor = self.transform(crop)

        row = self.df.iloc[idx]

        if pd.isna(row.x_1):
            x1, y1, x2, y2 = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            x1, y1, x2, y2 = int(row.x_1), int(row.y_1), int(row.x_2), int(row.y_2)

        return {
            self.input_tensors_key: image_tensor,
            self.labels_key: label,
            self.paths_key: row.path,
            self.x1_key: x1,
            self.y1_key: y1,
            self.x2_key: x2,
            self.y2_key: y2,
        }

    def __len__(self) -> int:
        return len(self.df)

    def read_image(self, idx: int) -> np.ndarray:
        img_bytes = self.read_bytes_image_cached(self.df.iloc[idx]["path"])
        img = self.f_imread(img_bytes)

        row = self.df.iloc[idx]

        if not pd.isna(row.x_1):
            x1, y1, x2, y2 = int(row.x_1), int(row.y_1), int(row.x_2), int(row.y_2)
            img = img[y1:y2, x1:x2, :]

        if isinstance(img, TImage):
            img = np.array(img)

        img = pad_resize(im=img, size=self.im_size, pad_ratio=self.pad_ratio)

        return img


class DatasetWithLabels(BaseDataset, IDatasetWithLabels):
    """
    The main purpose of this class is to be used as a dataset during
    the training stage.

    It has to know how to return its labels, which is required information
    to perform the training with the combinations-based losses.
    Particularly, these labels will be passed to Sampler to form the batches and
    batches will be passed to Miner to form the combinations.
    """

    def get_labels(self) -> np.ndarray:
        return np.array(self.df["label"].tolist())

    @property
    def num_labels(self) -> int:
        return self.df["label"].nunique()


class DatasetQueryGallery(BaseDataset, IDatasetQueryGallery):
    """
    The main purpose of this class is to be used as a dataset during
    the validation stage. It has to provide information
    about its query/gallery split.

    Note, that some of the datasets used as benchmarks in MetricLearning
    provide such information (for example, DeepFashion InShop), but some of them
    don't (for example, CARS196 or CUB200).
    The validation idea for the latter is to calculate the embeddings for the whole validation set,
    then for every item find top-k nearest neighbors and calculate the desired retrieval metric.
    In other words, for the desired query item, the gallery is the rest of the validation dataset.
    If you want to perform this kind of validation process, then simply return
    is_query == True and is_gallery == True for every item in the dataset.
    Note, that is_query and is_gallery can be True both at the same time. In this case, we perform
    a validation procedure for every item in the validation set using the "1 vs rest" approach.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        im_size: int,
        pad_ratio: float = 0.0,
        dataset_root: Optional[Union[str, Path]] = None,
        transform: Optional[albu.Compose] = None,
        f_imread: TImReader = imread_cv2,
        cache_size: int = 100_000,
        input_tensors_key: str = "input_tensors",
        labels_key: str = "labels",
        paths_key: str = "paths",
        x1_key: str = "x1",
        x2_key: str = "x2",
        y1_key: str = "y1",
        y2_key: str = "y2",
        is_query_key: str = "is_query",
        is_gallery_key: str = "is_gallery",
    ):
        super(DatasetQueryGallery, self).__init__(
            df=df,
            im_size=im_size,
            dataset_root=dataset_root,
            transform=transform,
            pad_ratio=pad_ratio,
            f_imread=f_imread,
            cache_size=cache_size,
            input_tensors_key=input_tensors_key,
            labels_key=labels_key,
            paths_key=paths_key,
            x1_key=x1_key,
            x2_key=x2_key,
            y1_key=y1_key,
            y2_key=y2_key,
        )
        assert all(x in df.columns for x in ("is_query", "is_gallery"))

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item[self.is_query_key] = bool(self.df.iloc[idx]["is_query"])
        item[self.is_gallery_key] = bool(self.df.iloc[idx]["is_gallery"])
        return item


def get_retrieval_datasets(
    dataset_root: Path,
    im_size_train: int,
    im_size_val: int,
    pad_ratio_train: float,
    pad_ratio_val: float,
    train_transform: Any,
    dataframe_name: str,
    cache_size: int = 100_000,
    use_label_encoder_for_train: bool = True,
) -> Tuple[DatasetWithLabels, DatasetQueryGallery]:
    df = pd.read_csv(dataset_root / dataframe_name, index_col=False)
    check_retrieval_dataframe_format(df, dataset_root=dataset_root)

    # train
    df_train = df[df["split"] == "train"].reset_index(drop=True)

    if use_label_encoder_for_train:
        mapper = {l: i for i, l in enumerate(df_train["label"].unique())}
        df_train["label"] = df_train["label"].map(mapper)

    train_dataset = DatasetWithLabels(
        df=df_train,
        dataset_root=dataset_root,
        im_size=im_size_train,
        pad_ratio=pad_ratio_train,
        transform=train_transform,
        cache_size=cache_size,
    )

    # val (query + gallery)
    df_query_gallery = df[df["split"] == "validation"].reset_index(drop=True)
    valid_dataset = DatasetQueryGallery(
        df=df_query_gallery,
        dataset_root=dataset_root,
        im_size=im_size_val,
        pad_ratio=pad_ratio_val,
        transform=None,
        cache_size=cache_size,
    )

    return train_dataset, valid_dataset


__all__ = ["BaseDataset", "DatasetWithLabels", "DatasetQueryGallery", "get_retrieval_datasets"]
