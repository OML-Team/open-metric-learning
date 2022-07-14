from typing import List, Union

import albumentations as albu
from albumentations.pytorch import ToTensorV2

from oml.const import TNormParam, MEAN, STD

TAugsList = List[Union[albu.ImageOnlyTransform, albu.DualTransform]]


def get_default_transforms_albu(mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    return albu.Compose([albu.Normalize(mean=mean, std=std), ToTensorV2()])
