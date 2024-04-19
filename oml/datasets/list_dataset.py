from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from torch.utils.data import Dataset

from oml.const import (
    INDEX_KEY,
    INPUT_TENSORS_KEY,
    LABELS_COLUMN,
    PATHS_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
    TBBoxes,
)
from oml.datasets.images import ImagesBaseDataset
from oml.transforms.images.torchvision import get_normalisation_torch
from oml.transforms.images.utils import TTransforms
from oml.utils.images.images import TImReader


class ListDataset(Dataset):
    # todo 522: remove the whole dataset

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
            f_imread: Function to read images, pass ``None`` so we pick it automatically based on provided transforms
            input_tensors_key: Key to put tensors into the batches
            cache_size: cache_size: Size of the dataset's cache
            index_key: Key to put samples' ids into the batches

        """
        data = defaultdict(list)
        data[PATHS_COLUMN] = list(map(str, filenames_list))
        data[LABELS_COLUMN] = ["none"] * len(filenames_list)

        if bboxes is not None:
            for bbox in bboxes:
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                else:
                    x1, y1, x2, y2 = None, None, None, None

                data[X1_COLUMN].append(x1)  # type: ignore
                data[Y1_COLUMN].append(y1)  # type: ignore
                data[X2_COLUMN].append(x2)  # type: ignore
                data[Y2_COLUMN].append(y2)  # type: ignore

        self._dataset = ImagesBaseDataset(
            paths=list(map(str, filenames_list)),
            bboxes=bboxes,
            transform=transform,
            f_imread=f_imread,
            input_tensors_key=input_tensors_key,
            cache_size=cache_size,
            index_key=index_key,
        )

        self.input_tensors_key = input_tensors_key
        self.index_key = index_key
        self.paths_key = self._dataset.paths_key

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._dataset[idx]

    def __len__(self) -> int:
        return len(self._dataset)


__all__ = ["ListDataset"]
