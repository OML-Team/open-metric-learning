from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from oml.exceptions import InvalidBBoxesException
from oml.transforms.images.torchvision.transforms import get_normalisation_torch
from oml.transforms.images.utils import TTransforms
from oml.utils.images.images import TImReader, imread_cv2

TBBox = Union[List[int], Tuple[int, int, int, int]]


class ListDataset(Dataset):
    """Iterate over list of images"""

    def __init__(
        self,
        filenames_list: Sequence[Path],
        bboxes: Optional[Dict[int, TBBox]] = None,
        transform: TTransforms = get_normalisation_torch(),
        f_imread: TImReader = imread_cv2,
        cache_size: int = 100_000,
    ):
        """
        Args:
            filenames_list: list of paths to images
            boxes: Sequences of bounding boxes. Should be either ``None`` or
                Sequence of of bboxes. If the image has multiple boxes, pass
                multiple image paths to ``filenames_list`` and for each
                filename provide one of its boxes. If you want to get
                embeddings for the whole image, set bbox to ``None`` for
                specific file.

                Format: x1, y1, x2, y2
            transform: torchvision or albumentations augmentations
            f_imread: function that opens image and returns bytes
            cache_size: cache_size: Size of the dataset's cache
        """
        self.filenames_list = filenames_list
        self.transform = transform
        self.f_imread = f_imread
        self.read_bytes_image_cached = lru_cache(maxsize=cache_size)(self._read_bytes_image)
        self.bboxes = bboxes or dict()

        self.validate_bboxes(bboxes, filenames_list)

    @staticmethod
    def validate_bboxes(bboxes: Optional[Dict[int, TBBox]], files: Sequence[Path]) -> None:
        if bboxes is not None:
            for idx, box in bboxes.items():
                if box is not None:
                    if len(box) != 4:
                        raise InvalidBBoxesException(f"Bbox size does not equal to 4: {box} for image {files[idx]}")
                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        raise InvalidBBoxesException(f"Bbox has invalid dimensions for image {files[idx]}")

    @staticmethod
    def _read_bytes_image(path: Union[Path, str]) -> bytes:
        with open(str(path), "rb") as fin:
            return fin.read()

    def crop(self, img: Union[Image.Image, np.ndarray], bbox: TBBox) -> np.ndarray:
        if isinstance(img, Image.Image):
            return np.array(img.crop(bbox))
        else:
            x1, y1, x2, y2 = bbox
            return img[y1:y2, x1:x2, :]

    def apply_transforms(self, img: Union[Image.Image], bbox: TBBox) -> torch.Tensor:
        img = self.crop(img, bbox)  # todo: since albu may handle bboxes we should move it to augs
        if self.transform is not None:
            if isinstance(self.transform, albu.Compose):
                image_tensor = self.transform(image=img)["image"]
            else:
                # torchvision.transforms
                image_tensor = self.transform(img)

        return image_tensor

    def __getitem__(self, idx: int) -> torch.Tensor:
        im_path = self.filenames_list[idx]
        img_bytes = self.read_bytes_image_cached(im_path)
        img = self.f_imread(img_bytes)
        bbox = self.bboxes.get(idx)

        if bbox is None:
            img_arr = np.array(img)
            print(img_arr.shape)
            im_h, im_w = img_arr.shape[:2]
            bbox = (0, 0, im_w, im_h)

        return self.apply_transforms(img, bbox)

    def __len__(self) -> int:
        return len(self.filenames_list)


__all__ = ["ListDataset"]
