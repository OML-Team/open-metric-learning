import multiprocessing as mp
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from albumentations.augmentations.functional import pad
except (AttributeError, ModuleNotFoundError, ImportError):
    from albumentations.augmentations.geometric.functional import pad

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from PIL.Image import Image as TPILImage

from oml.const import PAD_COLOR, TBBox, TColor

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


def draw_bbox(im: np.ndarray, bbox: Optional[TBBox], color: TColor) -> np.ndarray:
    """
    Draws a single bounding box on the image.
    If the elements of the bbox are NaNs, we will draw bbox around the whole image.

    Args:
        im: Image
        bbox: Optional single bounding box in the format of ``[x1, y1, x2, y2]``.
        color: Tuple of 3 ints
    """
    im_ret = im.copy()

    if bbox is None:
        x1, y1, x2, y2 = 0, 0, im_ret.shape[1], im_ret.shape[0]
    else:
        x1, y1, x2, y2 = bbox

    im_avg_sz = (im_ret.shape[0] + im_ret.shape[1]) / 2
    thickness = max(3, int(0.05 * im_avg_sz))
    im_ret = cv2.rectangle(im_ret, (x1, y1), (x2, y2), thickness=thickness, color=color)

    return im_ret


def square_pad(img: np.ndarray) -> np.ndarray:
    return pad(img, min_height=max(img.shape), min_width=max(img.shape), border_mode=0, value=PAD_COLOR)


def try_to_open_image(im_path: Path, f_imread: TImReader) -> Optional[str]:
    try:
        _ = f_imread(im_path)
        return None
    except Exception:
        return str(im_path)


def find_broken_images(images_list: List[Path], f_imread: TImReader, num_processes: int = 10) -> List[str]:
    try_to_open_image_ = partial(try_to_open_image, f_imread=f_imread)

    with mp.Pool(processes=num_processes) as p:
        results = list(tqdm(p.imap(try_to_open_image_, images_list), total=len(images_list)))

    results = list(filter(lambda x: x is not None, results))
    return results


def figure_to_nparray(fig: plt.Figure) -> np.ndarray:
    """
    Converts a matplotlib figure to a numpy array with RGB channels and no alpha channel.
    """
    fig.canvas.draw()
    data = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    plt.close(fig)
    return data


__all__ = [
    "TImage",
    "TImReader",
    "tensor_to_numpy_image",
    "imread_cv2",
    "imread_pillow",
    "draw_bbox",
    "square_pad",
    "figure_to_nparray",
]
