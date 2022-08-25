from typing import Optional

import albumentations as albu

from oml.registry.transforms import TTransforms
from oml.utils.images.images import TImReader, imread_cv2, imread_pillow


def get_im_reader_for_transforms(transforms: Optional[TTransforms]) -> TImReader:
    if (transforms is None) or isinstance(transforms, albu.Compose):
        return imread_cv2
    else:
        return imread_pillow
