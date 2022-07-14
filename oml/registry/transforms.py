from typing import Optional, Union

import albumentations as albu
import torchvision.transforms as t

from oml.const import MEAN, STD, TNormParam
from oml.transforms.images.albumentations.default import get_default_albu
from oml.transforms.images.albumentations.default_weak import get_default_weak_albu
from oml.transforms.images.albumentations.shared import get_normalisation_albu
from oml.transforms.images.torchvision.default import get_default_torch
from oml.transforms.images.torchvision.shared import get_normalisation_torch

TAugs = Union[albu.Compose, t.Compose]

AUGS_REGISTRY = {
    "default_albu": get_default_albu(),
    "default_weak_albu": get_default_weak_albu(),
    "default_torch": get_default_torch(),
}


def get_augs(name: str, mean: Optional[TNormParam] = MEAN, std: Optional[TNormParam] = STD) -> TAugs:
    augs = AUGS_REGISTRY[name]

    if isinstance(augs, albu.Compose):
        return albu.Compose([augs, get_normalisation_albu(mean=mean, std=std)])

    elif isinstance(augs, t.Compose):
        return t.Compose([get_normalisation_torch(mean=mean, std=std), augs])

    else:
        return augs
