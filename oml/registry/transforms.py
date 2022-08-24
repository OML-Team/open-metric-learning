from typing import Any, Dict, Union

import albumentations as albu
import torchvision.transforms as t

from oml.transforms.images.albumentations.default import get_default_albu
from oml.transforms.images.albumentations.default_weak import get_default_weak_albu
from oml.transforms.images.torchvision.default import get_default_torch
from oml.utils.misc import TCfg, dictconfig_to_dict

TTransforms = Union[albu.Compose, t.Compose]

TRANSFORMS_REGISTRY = {
    "default_albu": get_default_albu,
    "default_weak_albu": get_default_weak_albu,
    "default_torch": get_default_torch,
}


def get_transforms(name: str, **kwargs: Dict[str, Any]) -> TTransforms:
    augs = TRANSFORMS_REGISTRY[name](**kwargs)  # type: ignore
    return augs


def get_transforms_by_cfg(cfg: TCfg) -> TTransforms:
    cfg = dictconfig_to_dict(cfg)
    return get_transforms(name=cfg["name"], **cfg["args"])


__all__ = ["TTransforms", "TRANSFORMS_REGISTRY", "get_transforms", "get_transforms_by_cfg"]
