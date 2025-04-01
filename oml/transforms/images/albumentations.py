from typing import List, Union

import albumentations as albu
import cv2
from albumentations.pytorch import ToTensorV2

from oml.const import MEAN, MEAN_CLIP, PAD_COLOR, STD, STD_CLIP, TNormParam
from oml.utils.misc import adapt_argument_as_kwarg

TTransformsList = List[Union[albu.ImageOnlyTransform, albu.DualTransform]]


def get_normalisation_albu(mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    return albu.Compose([albu.Normalize(mean=mean, std=std), ToTensorV2()])


def get_normalisation_resize_albu(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    pad_value_kwarg = adapt_argument_as_kwarg(albu.PadIfNeeded, ["value", "fill"], PAD_COLOR)
    return albu.Compose(
        [
            albu.LongestMaxSize(max_size=im_size),
            albu.PadIfNeeded(min_height=im_size, min_width=im_size, border_mode=cv2.BORDER_CONSTANT, **pad_value_kwarg),
            albu.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_normalisation_resize_albu_clip(im_size: int) -> albu.Compose:
    return get_normalisation_resize_albu(im_size=im_size, mean=MEAN_CLIP, std=STD_CLIP)


__all__ = [
    "get_normalisation_albu",
    "get_normalisation_resize_albu",
    "get_normalisation_resize_albu_clip",
]
