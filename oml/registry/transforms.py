from typing import Any, Dict

from oml.transforms.images.albumentations.transforms import (
    get_augs_albu,
    get_normalisation_albu,
    get_normalisation_resize_albu,
    get_normalisation_resize_albu_clip,
)
from oml.transforms.images.torchvision.transforms import (
    get_augs_hypvit,
    get_augs_torch,
    get_normalisation_resize_hypvit,
    get_normalisation_resize_torch,
    get_normalisation_torch,
)
from oml.transforms.images.utils import TTransforms
from oml.utils.misc import TCfg, dictconfig_to_dict

TRANSFORMS_ALBU = {
    "augs_albu": get_augs_albu,
    "norm_albu": get_normalisation_albu,
    "norm_resize_albu": get_normalisation_resize_albu,
    "norm_resize_albu_clip": get_normalisation_resize_albu_clip,
}

TRANSFORMS_TORCH = {
    "augs_torch": get_augs_torch,
    "norm_torch": get_normalisation_torch,
    "norm_resize_torch": get_normalisation_resize_torch,
    "augs_hypvit_torch": get_augs_hypvit,
    "norm_resize_hypvit_torch": get_normalisation_resize_hypvit,
}

TRANSFORMS_REGISTRY = {**TRANSFORMS_ALBU, **TRANSFORMS_TORCH}


def get_transforms(name: str, **kwargs: Dict[str, Any]) -> TTransforms:
    augs = TRANSFORMS_REGISTRY[name](**kwargs)  # type: ignore
    return augs


def get_transforms_by_cfg(cfg: TCfg) -> TTransforms:
    cfg = dictconfig_to_dict(cfg)
    return get_transforms(name=cfg["name"], **cfg["args"])


__all__ = [
    "TRANSFORMS_TORCH",
    "TRANSFORMS_ALBU",
    "TRANSFORMS_REGISTRY",
    "get_transforms",
    "get_transforms_by_cfg",
]
