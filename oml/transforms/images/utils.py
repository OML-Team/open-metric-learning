from typing import Optional, Union

import albumentations as albu
import torchvision.transforms as t

from oml.utils.images.images import TImReader, imread_cv2, imread_pillow

TTransforms = Union[albu.Compose, t.Compose]


def get_im_reader_for_transforms(transforms: Optional[TTransforms]) -> TImReader:
    if isinstance(transforms, t.Compose):
        return imread_pillow
    else:
        return imread_cv2


__all__ = ["TTransforms", "get_im_reader_for_transforms"]
