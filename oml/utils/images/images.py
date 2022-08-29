from io import BytesIO
from pathlib import Path
from typing import Callable, Union

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from PIL.Image import Image as TPILImage

TImage = Union[PIL.Image.Image, np.ndarray]
TImReader = Callable[[Union[Path, str, bytes]], TImage]


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = img.detach().cpu()
    img = img.permute(1, 2, 0).numpy().copy()

    img -= img.min()
    img /= img.max()
    img *= 255

    img = img.astype(np.uint8)

    return img


def imread_cv2(im_src: Union[Path, str, bytes]) -> np.ndarray:
    if isinstance(im_src, (Path, str)):
        image = cv2.imread(str(im_src), cv2.IMREAD_UNCHANGED)
    elif isinstance(im_src, bytes):
        image_raw = np.frombuffer(im_src, np.uint8)
        image = cv2.imdecode(image_raw, cv2.IMREAD_UNCHANGED)
    else:
        raise TypeError("Unsupported type")

    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)


def imread_pillow(im_src: Union[Path, str, bytes]) -> TPILImage:
    if isinstance(im_src, (Path, str)):
        image = Image.open(im_src)
    elif isinstance(im_src, bytes):
        image = Image.open(BytesIO(im_src))
    else:
        raise TypeError("Unsupported type")
    return image.convert("RGB")


__all__ = [
    "TImage",
    "TImReader",
    "tensor_to_numpy_image",
    "imread_cv2",
    "imread_pillow",
]
