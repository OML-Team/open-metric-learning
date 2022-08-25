from typing import Optional

import torchvision.transforms as t

from oml.registry.transforms import TTransforms
from oml.utils.images.images import TImReader, imread_cv2, imread_pillow


def get_im_reader_for_transforms(transforms: Optional[TTransforms]) -> TImReader:
    if isinstance(transforms, t.Compose):
        return imread_pillow
    else:
        return imread_cv2


__all__ = ["get_im_reader_for_transforms"]
