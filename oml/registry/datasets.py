from typing import Dict, Any, Type, Tuple

import pandas as pd

from oml.const import TCfg, LABELS_COLUMN, SPLIT_COLUMN
from oml.interfaces.datasets import IBaseDataset
from oml.datasets.images import ImageQueryGalleryLabeledDataset, ImageQueryGalleryDataset, get_retrieval_images_datasets
from oml.registry.transforms import get_transforms_by_cfg

def _build_image_dataset(**cfg) -> Tuple[ImageQueryGalleryDataset, ImageQueryGalleryLabeledDataset]:
    transforms_train = get_transforms_by_cfg(cfg["transforms_train"])
    transforms_val = get_transforms_by_cfg(cfg["transforms_val"])

    dataset_train, dataset_val = get_retrieval_images_datasets(
        dataset_root=cfg["dataset_root"],
        dataframe_name=cfg["df"],
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        cache_size=cfg["cache_size"],
        verbose=cfg.get("verbose", True),
        )

    return dataset_train, dataset_val

def build_image_dataset(dataset_name: str, **kwargs):
    return DATASET_BUILDER_REGISTRY[dataset_name](**kwargs)

DATASET_BUILDER_REGISTRY = {
    "oml_image_dataset": _build_image_dataset,
    }







