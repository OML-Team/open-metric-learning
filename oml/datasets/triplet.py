import logging
import warnings
from functools import lru_cache
from itertools import chain
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Tuple, Union

import albumentations as albu
import numpy as np
import torch
from torch.utils.data import Dataset

from oml.const import INPUT_TENSORS_KEY
from oml.transforms.images.albumentations.transforms import get_normalisation_albu
from oml.utils.images.images import TImReader, imread_cv2

TPath = Union[Path, str]
TTriplet = Tuple[TPath, TPath, TPath]
TItem = Dict[str, Any]


class TriDataset(Dataset):
    def __init__(
        self,
        triplets: List[TTriplet],
        im_root: Path,
        transforms: albu.Compose,
        expand_ratio: float,
        f_imread: TImReader = imread_cv2,
        cache_size: int = 50_000,
    ):
        """

        Args:
            triplets: List of triplets
            im_root: Images directory
            transforms: Image transforms
            expand_ratio: Set expand_ratio > 0 to generate additional triplets.
                          We keep positive pairs, but generale negative ones randomly.
                          After this procedure you dataset's length will increased (1 + expand_ratio) times
            f_imread: Function to read image from disk

        """
        assert expand_ratio >= 0

        warnings.warn("For now we do not keep this dataset up to date. Use it carefully.")

        self.triplets = triplets

        self.all_ims = set(chain(*triplets))

        self.im_root = im_root
        self.expand_ratio = expand_ratio
        self.f_imread = f_imread

        self.transforms = transforms or get_normalisation_albu()
        assert isinstance(transforms, albu.Compose) or (transforms is None)

        self.read_bytes_image_cached = lru_cache(maxsize=cache_size)(self._read_bytes_image)

        logging.info(f"Dataset contains {len(self.triplets)} triplets.")

    @staticmethod
    def _read_bytes_image(path: Union[Path, str]) -> bytes:
        with open(str(path), "rb") as fin:
            return fin.read()

    def get_image(self, path: Union[Path, str]) -> np.ndarray:
        image_bytes = self.read_bytes_image_cached(path)
        image = self.f_imread(image_bytes)
        return image

    def __len__(self) -> int:
        return int((1 + self.expand_ratio) * len(self.triplets))

    def __getitem__(self, idx: int) -> TItem:  # type: ignore
        if idx < len(self.triplets):
            triplet = self.triplets[idx]
        else:
            # here we randomly create negative pair for the picked positive one
            a, p, _ = sample(self.triplets, k=1)[0]
            n = sample(self.all_ims - {a, p}, k=1)[0]
            triplet = a, p, n

        assert len(triplet) == 3

        images = tuple(map(lambda x: self.get_image(self.im_root / x), triplet))

        tensors = tuple(map(lambda x: self.transforms(image=x)["image"], images))

        tri_ids = (f"{idx}_a", f"{idx}_p", f"{idx}_n")
        return {INPUT_TENSORS_KEY: tensors, "tri_ids": tri_ids, "images": images}


def tri_collate(items: List[TItem]) -> Dict[str, Any]:
    batch = dict()

    for key in (INPUT_TENSORS_KEY,):
        batch[key] = torch.stack(list(chain(*[item[key] for item in items])))

    for key in ("tri_ids", "images"):
        if key in items[0].keys():
            batch[key] = list(chain(*[item[key] for item in items]))

    return batch


__all__ = ["TPath", "TTriplet", "TItem", "TriDataset", "tri_collate"]
