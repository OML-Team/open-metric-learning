from typing import Optional

import albumentations as albu

from oml.const import MEAN, STD, TNormParam
from oml.utils.images.augs import get_all_augs, get_default_transforms_albu
from oml.utils.images.augs_weak import get_all_augs_weak

AUGS_REGISTRY = {"all": get_all_augs(), "all_weak": get_all_augs_weak(), "no_augs": albu.Compose([])}


def get_augs(name: str) -> albu.Compose:
    return AUGS_REGISTRY[name]


def get_augs_with_default(
    name: str, mean: Optional[TNormParam] = MEAN, std: Optional[TNormParam] = STD
) -> albu.Compose:
    return albu.Compose([AUGS_REGISTRY[name], get_default_transforms_albu(mean=mean, std=std)])
