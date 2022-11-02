from io import BytesIO
from pathlib import Path
from typing import Callable, Union

import albumentations as albu
import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from PIL.Image import Image as TPILImage

from oml.const import PAD_COLOR, TColor

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


def draw_bbox(im: np.ndarray, bbox: torch.Tensor, color: TColor) -> np.ndarray:
    """
    Draws a single bounding box on the image.
    If the elements of the bbox are NaNs, we will draw bbox around the whole image.

    Args:
        im: Image
        bbox: Single bounding in the format of [x1, y1, x2, y2]
        color: Tuple of 3 ints
    """
    im_ret = im.copy()
    if not any(torch.isnan(bbox)):
        x1, y1, x2, y2 = list(map(int, bbox))
    elif all(torch.isnan(bbox)):
        x1, y1, x2, y2 = 0, 0, im_ret.shape[1], im_ret.shape[0]
    else:
        raise ValueError("BBox can only consist of all NaNs or all numbers.")

    im_ret = cv2.rectangle(im_ret, (x1, y1), (x2, y2), thickness=15, color=color)

    return im_ret


def get_img_with_bbox(im_path: str, bbox: torch.Tensor, color: TColor) -> np.ndarray:
    """
    Reads the image by its name and draws bbox on it.

    Args:
        im_path: Image path
        bbox: Single bounding box in the format of [x1, y1, x2, y2]. It may also be a list of 4 torch("nan").
        color: Tuple of 3 ints from 0 to 255
    """
    img = imread_cv2(im_path)
    img = draw_bbox(img, bbox, color)
    return img


def square_pad(img: np.ndarray) -> np.ndarray:
    return albu.functional.pad(img, min_height=max(img.shape), min_width=max(img.shape), border_mode=0, value=PAD_COLOR)


__all__ = [
    "TImage",
    "TImReader",
    "tensor_to_numpy_image",
    "imread_cv2",
    "imread_pillow",
    "draw_bbox",
    "get_img_with_bbox",
    "square_pad",
]
