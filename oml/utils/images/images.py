from typing import Callable, Union

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as TImage

TImReader = Callable[[str], Union[np.ndarray, TImage]]


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu()
    img = img.permute(1, 2, 0).numpy().copy()

    img -= img.min()
    img /= img.max()
    img *= 255

    img = img.astype(np.uint8)

    return img


def imread_cv2(im_path: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(im_path)), cv2.COLOR_BGRA2RGB)
    if img is None:
        raise ValueError("Image can not be read")
    else:
        return img


def imread_pillow(im_path: str) -> TImage:
    return Image.open(im_path)


__all__ = ["TImReader", "tensor_to_numpy_image", "imread_cv2", "imread_pillow"]
