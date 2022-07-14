from typing import Any, Optional, Union

import albumentations as albu
import torchvision.transforms as t

from oml.const import MEAN, STD, TNormParam
from oml.transforms.images.albumentations.default import get_default_albu
from oml.transforms.images.albumentations.default_weak import get_default_weak_albu
from oml.transforms.images.albumentations.shared import get_normalisation_albu
from oml.transforms.images.torchvision.default import get_default_torch
from oml.transforms.images.torchvision.shared import get_normalisation_torch

AUGS_REGISTRY = {
    "default_albu": get_default_albu(),
    "default_weak_albu": get_default_weak_albu(),
    "no_augs": albu.Compose([]),
    "default_torch": get_default_torch(),
}


def get_augs(
    name: str, mean: Optional[TNormParam] = MEAN, std: Optional[TNormParam] = STD
) -> Union[albu.Compose, t.Compose, Any]:
    augs = AUGS_REGISTRY[name]

    if isinstance(augs, albu.Compose):
        return albu.Compose([augs, get_normalisation_albu(mean=mean, std=std)])

    elif isinstance(augs, t.Compose):
        return t.Compose([augs, get_normalisation_torch(mean=mean, std=std)])

    else:
        return augs
