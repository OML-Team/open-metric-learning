from typing import List, Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from oml.const import MEAN, STD, TNormParam

TTransformsList = List[Union[albu.ImageOnlyTransform, albu.DualTransform]]


def get_normalisation_albu(mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    return albu.Compose([albu.Normalize(mean=mean, std=std), ToTensorV2()])


__all__ = ["TTransformsList", "get_normalisation_albu"]
