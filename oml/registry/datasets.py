from pathlib import Path
from typing import Any, Dict, Tuple

from oml.const import TCfg
from oml.datasets.images import (
    ImageQueryGalleryDataset,
    ImageQueryGalleryLabeledDataset,
    get_retrieval_images_datasets,
)
from oml.registry.transforms import get_transforms_by_cfg


def _build_image_dataset(**cfg: TCfg) -> Tuple[ImageQueryGalleryDataset, ImageQueryGalleryLabeledDataset]:
    transforms_train = get_transforms_by_cfg(cfg["transforms_train"])
    transforms_val = get_transforms_by_cfg(cfg["transforms_val"])

    dataset_train, dataset_val = get_retrieval_images_datasets(
        dataset_root=Path(str(cfg["dataset_root"])),
        dataframe_name=str(cfg["df"]),
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        cache_size=int(str(cfg["cache_size"])),
        verbose=bool(cfg.get("verbose", True)),
    )

    return dataset_train, dataset_val


def build_image_dataset(
    dataset_name: str, **kwargs: Dict[str, Any]
) -> Tuple[ImageQueryGalleryDataset, ImageQueryGalleryLabeledDataset]:
    return DATASET_BUILDER_REGISTRY[dataset_name](**kwargs)


DATASET_BUILDER_REGISTRY = {
    "oml_image_dataset": _build_image_dataset,
}

__all__ = ["build_image_dataset", "DATASET_BUILDER_REGISTRY"]
