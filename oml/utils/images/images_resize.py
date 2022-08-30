from functools import partial
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch

SizeHW = Tuple[int, int]  # Size of images in (height, width) order
BboxesType = Union[torch.Tensor, np.ndarray]  # in format Nx4: [[left_1, top_1, right_1, bottom_1], ...]
ImageType = Union[torch.Tensor, np.ndarray]  # ndarray shapes: HxW, HxWxC. tensor shapes : HxW, CxHxW, BxCxHxW


def karesize_image(image: ImageType, new_hw: SizeHW) -> ImageType:
    """
    Function resizes image with keeping aspect ratio

    NOTE: tensors are resized on same device with gradient calculation

    """
    if isinstance(image, torch.Tensor):
        is_tensor = True
    elif isinstance(image, np.ndarray):
        is_tensor = False
    else:
        raise TypeError("Unsupported type")

    src_hw = get_image_hw(image)

    if new_hw == src_hw:
        return image

    _, resize_hw, padding = get_karesize_param(src_hw, new_hw)

    if is_tensor:
        image = resize_tensor(image, resize_hw)
        image = torch.nn.functional.pad(image, padding, mode="constant")
    else:
        pad_left, pad_right, pad_top, pad_bottom = padding
        padding_np = ((pad_top, pad_bottom), (pad_left, pad_right))
        if image.ndim == 3:
            padding_np = (*padding_np, (0, 0))  # type: ignore
        image = resize_ndarray(image, resize_hw)
        image = np.pad(image, padding_np)

    return image


def inverse_karesize_image(image: ImageType, src_hw: SizeHW) -> ImageType:
    """
    Function resizes image back to src size. The function inverse to the 'karesize_image' function

    NOTE: tensors are resized on same device with gradient calculation

    """
    if isinstance(image, torch.Tensor):
        is_tensor = True
    elif isinstance(image, np.ndarray):
        is_tensor = False
    else:
        raise TypeError("Unsupported type")

    curr_hw = get_image_hw(image)

    if curr_hw == src_hw:
        return image

    _, (resized_h, resized_w), (pad_left, _, pad_top, _) = get_karesize_param(src_hw, curr_hw)

    if is_tensor:
        image = image[..., pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
        image = resize_tensor(image, src_hw)
    else:
        image = image[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w, ...]
        image = resize_ndarray(image, src_hw)

    return image


def karesize_bboxes(src_bboxes: BboxesType, src_hw: SizeHW, new_hw: SizeHW) -> BboxesType:
    """
    Function resizes bboxes with keeping aspect ratio from src size to desired size

    NOTE: tensors are resized on same device with gradient calculation

    """
    _assert_bbox_type_and_shape(src_bboxes)
    scale, _, (pad_left, _, pad_top, _) = get_karesize_param(src_hw, new_hw)

    bboxes = src_bboxes * scale
    bboxes[:, 0] += pad_left
    bboxes[:, 2] += pad_left
    bboxes[:, 1] += pad_top
    bboxes[:, 3] += pad_top

    return bboxes


def inverse_karesize_bboxes(cur_bboxes: BboxesType, curr_hw: SizeHW, src_hw: SizeHW) -> BboxesType:
    """
    Function resizes bboxes back to src size. The function inverse to the 'karesize_bboxes' function

    NOTE: tensors are resized on same device with gradient calculation

    """
    _assert_bbox_type_and_shape(cur_bboxes)
    scale, _, (pad_left, _, pad_top, _) = get_karesize_param(src_hw, curr_hw)

    padding = [[pad_left, pad_top, pad_left, pad_top]]

    if isinstance(cur_bboxes, np.ndarray):
        constructor = np.array
    elif isinstance(cur_bboxes, torch.Tensor):
        constructor = partial(torch.tensor, device=cur_bboxes.device)
    else:
        raise TypeError("Unsuported type of bboxes")

    padding = constructor(padding)

    src_bboxes = (cur_bboxes - padding) / scale

    return src_bboxes


def _assert_bbox_type_and_shape(cur_bboxes: BboxesType) -> None:
    assert isinstance(cur_bboxes, (torch.Tensor, np.ndarray))
    assert cur_bboxes.ndim == 2
    assert cur_bboxes.shape[1] == 4


def get_image_hw(image: Union[np.ndarray, torch.Tensor]) -> SizeHW:
    if isinstance(image, np.ndarray):
        image_hw = (image.shape[0], image.shape[1])
    elif isinstance(image, torch.Tensor):
        image_hw = (image.shape[-2], image.shape[-1])
    else:
        raise TypeError("Only array and tensor types are available")
    return image_hw


def get_karesize_param(src_hw: SizeHW, new_hw: SizeHW) -> Tuple[float, SizeHW, List[int]]:
    scale = min(new_hw[0] / src_hw[0], new_hw[1] / src_hw[1])
    resized_hw = (int(scale * src_hw[0]), int(scale * src_hw[1]))

    pad_left = (new_hw[1] - resized_hw[1]) // 2
    pad_right = new_hw[1] - resized_hw[1] - pad_left

    pad_top = (new_hw[0] - resized_hw[0]) // 2
    pad_bottom = new_hw[0] - resized_hw[0] - pad_top

    return scale, resized_hw, [pad_left, pad_right, pad_top, pad_bottom]


def resize_tensor(tensor: torch.Tensor, new_size_hw: SizeHW) -> torch.Tensor:
    assert isinstance(tensor, torch.Tensor)
    ndim = tensor.ndim
    assert 2 <= ndim <= 4

    dtype = tensor.dtype

    # Tensor must have BxCxHxW shape to interpolate its in spatial dimension
    for _ in range(4 - ndim):
        tensor = tensor.unsqueeze(0)

    min_value = torch.min(tensor).item()
    max_value = torch.max(tensor).item()

    tensor_resized = torch.nn.functional.interpolate(
        tensor.float(), (new_size_hw[0], new_size_hw[1]), align_corners=True, mode="bicubic"
    ).clamp(min_value, max_value)
    for _ in range(4 - ndim):
        tensor_resized = tensor_resized.squeeze(0)

    if dtype == torch.bool:
        if tensor.requires_grad:
            print(
                "Resizing for torch.bool is not implemented in pytorch. " "Gradients will be discarded due to resizing",
                flush=True,
            )
        thr = (max_value - min_value) / 2
        tensor_resized = tensor_resized >= thr

    tensor_resized = tensor_resized.to(dtype=dtype)

    return tensor_resized


def resize_ndarray(array: np.ndarray, new_size_hw: SizeHW) -> np.ndarray:
    assert isinstance(array, np.ndarray)
    ndim = array.ndim
    assert 2 <= ndim <= 3

    dtype = array.dtype

    if dtype == bool:
        array = array.astype(np.uint8)

    array_resized = cv2.resize(array, (new_size_hw[1], new_size_hw[0]), interpolation=cv2.INTER_CUBIC).astype(dtype)

    if array_resized.ndim < ndim:
        array_resized = array_resized[..., np.newaxis]

    return array_resized


__all__ = [
    "karesize_image",
    "inverse_karesize_image",
    "karesize_bboxes",
    "inverse_karesize_bboxes",
    "get_image_hw",
    "get_karesize_param",
    "resize_tensor",
    "resize_ndarray",
]
