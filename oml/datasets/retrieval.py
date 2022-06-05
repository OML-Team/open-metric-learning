from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import albumentations as albu
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from oml.interfaces.datasets import IDatasetQueryGallery, IDatasetWithLabels
from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.images.augs import get_default_transforms_albu
from oml.utils.images.images import TImReader, imread_cv2
from oml.utils.images.images_resize import pad_resize


class BaseDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            im_size: int,
            pad_ratio: float,
            dataset_root: Optional[Path] = None,
            transform: Optional[albu.Compose] = None,
            f_imread: TImReader = imread_cv2,
    ):
        assert pad_ratio >= 0
        assert all(x in df.columns for x in ("label", "path"))

        if not all(coord in df.columns for coord in ("x_1", "x_2", "y_1", "y_2")):
            df["x_1"] = None
            df["x_2"] = None
            df["y_1"] = None
            df["y_2"] = None

        if dataset_root is not None:
            df["path"] = df["path"].apply(lambda x: str(dataset_root / x))

        self.df = df
        self.im_size = im_size
        self.pad_ratio = pad_ratio
        self.transform = transform or get_default_transforms_albu()
        self.f_imread = f_imread

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        crop = self.read_image(idx)
        image_tensor = self.transform(image=crop)["image"]
        label = self.df.iloc[idx]["label"]

        row = self.df.iloc[idx]

        if pd.isna(row.x_1):
            x1, y1, x2, y2 = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            x1, y1, x2, y2 = int(row.x_1), int(row.y_1), int(row.x_2), int(row.y_2)

        return {
            "input_tensors": image_tensor,
            "labels": label,
            "paths": row.path,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    def __len__(self) -> int:
        return len(self.df)

    @lru_cache(maxsize=100_000)
    def read_image(self, idx: int) -> np.ndarray:
        img = self.f_imread(self.df.iloc[idx]["path"])

        row = self.df.iloc[idx]

        if not pd.isna(row.x_1):
            x1, y1, x2, y2 = int(row.x_1), int(row.y_1), int(row.x_2), int(row.y_2)
            img = img[y1:y2, x1:x2, :]

        img = pad_resize(im=img, size=self.im_size, pad_ratio=self.pad_ratio)

        return img


class DatasetWithLabels(BaseDataset, IDatasetWithLabels):
    def get_labels(self) -> np.ndarray:
        return np.array(self.df["label"].tolist())


class DatasetQueryGallery(BaseDataset, IDatasetQueryGallery):
    def __init__(
            self,
            df: pd.DataFrame,
            im_size: int,
            pad_ratio: float,
            dataset_root: Optional[Path] = None,
            transform: Optional[albu.Compose] = None,
            f_imread: TImReader = imread_cv2,
    ):
        super(DatasetQueryGallery, self).__init__(
            df=df,
            im_size=im_size,
            dataset_root=dataset_root,
            transform=transform,
            pad_ratio=pad_ratio,
            f_imread=f_imread,
        )
        assert all(x in df.columns for x in ("is_query", "is_gallery"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item["is_query"] = bool(self.df.iloc[idx]["is_query"])
        item["is_gallery"] = bool(self.df.iloc[idx]["is_gallery"])
        return item


def get_retrieval_datasets(
        dataset_root: Path,
        im_size: int,
        pad_ratio_train: float,
        pad_ratio_val: float,
        train_transform: Any,
        dataframe_name: str = "df.csv"
) -> Tuple[DatasetWithLabels, DatasetQueryGallery]:
    df = pd.read_csv(dataset_root / dataframe_name, index_col=False)
    check_retrieval_dataframe_format(df, dataset_root=dataset_root)

    # train
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    train_dataset = DatasetWithLabels(
        df=df_train, dataset_root=dataset_root, im_size=im_size, pad_ratio=pad_ratio_train, transform=train_transform
    )

    # val (query + gallery)
    df_query_gallery = df[df["split"] == "validation"].reset_index(drop=True)
    valid_dataset = DatasetQueryGallery(
        df=df_query_gallery, dataset_root=dataset_root, im_size=im_size, pad_ratio=pad_ratio_val
    )

    return train_dataset, valid_dataset
