from pathlib import Path
from typing import Any, Dict, Tuple

from oml.const import TCfg
from oml.datasets.images import get_retrieval_images_datasets
from oml.interfaces.datasets import ILabeledDataset, IQueryGalleryLabeledDataset
from oml.registry.transforms import get_transforms_by_cfg


def get_image_datasets_by_cfg(**cfg: TCfg) -> Tuple[ILabeledDataset, IQueryGalleryLabeledDataset]:

    transforms_train = cfg.get("transforms_train", None)
    transforms_val = cfg.get("transforms_val", None)

    if transforms_train is not None:
        transforms_train = get_transforms_by_cfg(transforms_train)

    if transforms_val is not None:
        transforms_val = get_transforms_by_cfg(transforms_val)

    dataset_train, dataset_val = get_retrieval_images_datasets(
        dataset_root=Path(str(cfg["dataset_root"])),
        dataframe_name=str(cfg["dataframe_name"]),
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        cache_size=int(str(cfg["cache_size"])),
        verbose=bool(cfg.get("verbose", True)),
    )

    return dataset_train, dataset_val


def get_image_datasets(
    dataset_name: str, **kwargs: Dict[str, Any]
) -> Tuple[ILabeledDataset, IQueryGalleryLabeledDataset]:
    return DATASET_BUILDER_REGISTRY[dataset_name](**kwargs)


DATASET_BUILDER_REGISTRY = {
    "oml_image_dataset": get_image_datasets_by_cfg,
}

__all__ = ["get_image_datasets", "DATASET_BUILDER_REGISTRY"]
