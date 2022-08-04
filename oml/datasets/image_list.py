from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from albumentations.core.composition import Compose as AlbuCompose
from torch.utils.data import Dataset
from torchvision.transforms import Compose as TorchvisionCompose

from oml.utils.images.images import TImReader, imread_cv2
from oml.utils.images.images_resize import pad_resize


class ImageListDataset(Dataset):
    def __init__(
        self,
        im_paths: List[Union[str, Path]],
        image_size: int,
        pad_ratio: float,
        transforms: Union[AlbuCompose, TorchvisionCompose],
        f_imread: TImReader = imread_cv2,
        cache_size: int = 100_000,
    ):
        """

        Args:
            im_paths: Paths to images
            image_size: Images will be resized to (image_size, image_size)
            transforms: Image transforms
            f_imread: Function to read image from disk

        """
        assert pad_ratio >= 0

        self.im_paths = im_paths
        self.image_size = image_size
        self.pad_ratio = pad_ratio
        self.transform = transforms
        self.f_imread = f_imread
        self.load_png_image_cached = lru_cache(cache_size)(self.load_png_image)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        path = self.im_paths[item]
        image = self.load_png_image_cached(im_path=path)

        if isinstance(self.transform, AlbuCompose):
            image = self.transform(image=image)["image"]
        else:
            image = self.transform(image)

        return image, Path(path).name

    def __len__(self) -> int:
        return len(self.im_paths)

    def load_png_image(self, im_path: Path) -> np.ndarray:
        image = self.f_imread(str(im_path))
        image = pad_resize(im=image, size=self.image_size, pad_ratio=self.pad_ratio)
        return image


__all__ = ["ImageListDataset"]
