import inspect
from pathlib import Path
from typing import Optional

import pandas as pd

from oml.const import LABELS_COLUMN, SPLIT_COLUMN, TCfg
from oml.datasets.images import (
    ImageLabeledDataset,
    ImageQueryGalleryDataset,
    ImageQueryGalleryLabeledDataset,
)
from oml.interfaces.datasets import IBaseDataset
from oml.utils.dataframe_format import check_retrieval_dataframe_format

DATASETS_REGISTRY = {
    "image_labeled_dataset": ImageLabeledDataset,
    "image_query_gallery_labeled_dataset": ImageQueryGalleryLabeledDataset,
    "image_query_gallery_dataset": ImageQueryGalleryDataset,
}


def get_dataset_by_cfg(cfg: TCfg, split: Optional[str] = None) -> IBaseDataset:
    if split and "dataframe_name" in cfg["args"].keys():
        df = pd.read_csv(Path(cfg["args"]["dataset_root"]) / cfg["args"]["dataframe_name"], index_col=False)

        check_retrieval_dataframe_format(
            df,
            dataset_root=Path(cfg["args"]["dataset_root"]),
            verbose=cfg["args"].get("show_dataset_warnings", True),
        )

        mapper = {l: i for i, l in enumerate(df.sort_values(by=[SPLIT_COLUMN])[LABELS_COLUMN].unique())}

        df = df[df[SPLIT_COLUMN] == split].reset_index(drop=True)
        df[LABELS_COLUMN] = df[LABELS_COLUMN].map(mapper)
        cfg["args"]["df"] = df

    dataset_class = DATASETS_REGISTRY[cfg["name"]]
    expected_args = inspect.signature(dataset_class.__init__).parameters  # type: ignore
    filtered_args = {key: cfg["args"][key] for key in cfg["args"] if key in expected_args}

    return dataset_class(**filtered_args)
