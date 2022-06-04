from typing import Callable

import cv2
import numpy as np
import torch

TImReader = Callable[[str], np.ndarray]


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
