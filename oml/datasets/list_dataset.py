from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import albumentations as albu
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from oml.const import INDEX_KEY, INPUT_TENSORS_KEY
from oml.exceptions import InvalidBBoxesException
from oml.transforms.images.torchvision.transforms import get_normalisation_torch
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader

TBBox = Tuple[int, int, int, int]
TBBoxes = Sequence[Optional[TBBox]]


class ListDataset(Dataset):
    """This is a dataset to iterate over a list of images."""

    def __init__(
        self,
        filenames_list: Sequence[Path],
        bboxes: Optional[TBBoxes] = None,
        transform: TTransforms = get_normalisation_torch(),
        f_imread: Optional[TImReader] = None,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        cache_size: Optional[int] = 0,
        index_key: str = INDEX_KEY,
    ):
        """
        Args:
            filenames_list: list of paths to images
            bboxes: Should be either ``None`` or a sequence of bboxes.
                If an image has ``N`` boxes, duplicate its
                path ``N`` times and provide bounding box for each of them.
                If you want to get an embedding for the whole image, set bbox to ``None`` for
                this particular image path. The format is ``x1, y1, x2, y2``.
            transform: torchvision or albumentations augmentations
            f_imread: Function to read the images, pass ``None`` so we pick it autmatically based on provided transforms
            input_tensors_key: Key to put tensors into the batches
            cache_size: cache_size: Size of the dataset's cache
            index_key: Key to put samples' ids into the batches

        """
        self.filenames_list = filenames_list
        self.transform = transform
        self.f_imread = f_imread or get_im_reader_for_transforms(transform)
        self.read_bytes_image = (
            lru_cache(maxsize=cache_size)(self._read_bytes_image) if cache_size else self._read_bytes_image
        )
        self.bboxes = bboxes

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key

        self.validate_bboxes(bboxes, filenames_list)

    @staticmethod
    def validate_bboxes(bboxes: Optional[TBBoxes], files: Sequence[Path]) -> None:
        if bboxes is not None:
            if len(bboxes) != len(files):
                raise InvalidBBoxesException(f"Number of boxes and files missmatch: {len(bboxes)=} != {len(files)}")
            for box, file_ in zip(bboxes, files):
                if box is not None:
                    if len(box) != 4:
                        raise InvalidBBoxesException(f"Bbox size does not equal to 4: {box} for image {file_}")
                    x1, y1, x2, y2 = box
                    if any(coord < 0 for coord in box):
                        raise InvalidBBoxesException(f"Bbox coordintates cannot be negative. File: {file_}")
                    if x2 < x1 or y2 < y1:
                        raise InvalidBBoxesException(f"Bbox has invalid dimensions for image {file_}")

    @staticmethod
    def _read_bytes_image(path: Union[Path, str]) -> bytes:
        with open(str(path), "rb") as fin:
            return fin.read()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        im_path = self.filenames_list[idx]
        img_bytes = self.read_bytes_image(im_path)  # type: ignore
        img = self.f_imread(img_bytes)
        if self.bboxes is not None:
            bbox = self.bboxes[idx]
        else:
            bbox = None

        if bbox is None:
            im_h, im_w = img.shape[:2] if isinstance(img, np.ndarray) else img.size[::-1]
            bbox = (0, 0, im_w, im_h)

        if isinstance(img, Image.Image):
            img = img.crop(bbox)
        else:
            x1, y1, x2, y2 = bbox
            img = img[y1:y2, x1:x2, :]

        if isinstance(self.transform, albu.Compose):
            image_tensor = self.transform(image=img)["image"]
        else:
            # torchvision.transforms
            image_tensor = self.transform(img)

        return {self.input_tensors_key: image_tensor, self.index_key: idx}

    def __len__(self) -> int:
        return len(self.filenames_list)


__all__ = ["ListDataset"]
