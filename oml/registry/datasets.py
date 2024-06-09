from typing import Dict, Any, Type

from oml.const import TCfg
from oml.interfaces.datasets import IBaseDataset
from oml.datasets.images import ImageQueryGalleryLabeledDataset, ImageQueryGalleryDataset

DATASETS_REGISTRY: Dict[str, Type[IBaseDataset]] = {
    'image_qg_labeled_dataset': ImageQueryGalleryLabeledDataset,
    }


def _get_dataset(registry: Dict[str, Any], dataset_name: str, **kwargs) -> IBaseDataset:
    dataset = registry[dataset_name](**kwargs)
    return dataset


def get_dataset(dataset_name: str, **kwargs: Dict[str, Any]) -> IBaseDataset:
    return _get_dataset(dataset_name=dataset_name,
                        registry=DATASETS_REGISTRY,
                        **kwargs)


__all__ = [
    "DATASETS_REGISTRY",
    "get_dataset",
    ]