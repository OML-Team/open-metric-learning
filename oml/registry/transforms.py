from typing import Optional

import albumentations as albu

from oml.const import MEAN, STD, TNormParam
from oml.transforms.images.albumentations.default import get_all_augs
from oml.transforms.images.albumentations.default_weak import get_all_augs_weak
from oml.transforms.images.albumentations.shared import get_default_transforms_albu

AUGS_REGISTRY = {
    "default_albu": get_all_augs(),
    "default_weak_albu": get_all_augs_weak(),
    "no_augs": albu.Compose([])
}


def get_augs(name: str) -> albu.Compose:
    return AUGS_REGISTRY[name]


def get_augs_with_default(
        name: str, mean: Optional[TNormParam] = MEAN, std: Optional[TNormParam] = STD
) -> albu.Compose:
    return albu.Compose([AUGS_REGISTRY[name], get_default_transforms_albu(mean=mean, std=std)])
