from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from oml.const import INDEX_KEY, PAIR_1ST_KEY, PAIR_2ND_KEY
from oml.datasets.list_dataset import ListDataset, TBBoxes
from oml.interfaces.datasets import IPairsDataset
from oml.transforms.images.torchvision.transforms import get_normalisation_torch
from oml.transforms.images.utils import TTransforms
from oml.utils.images.images import TImReader, imread_pillow


class EmbeddingPairsDataset(IPairsDataset):
    """
    Dataset to iterate over pairs of embeddings.

    """

    def __init__(
        self,
        embeddings1: Tensor,
        embeddings2: Tensor,
        pair_1st_key: str = PAIR_1ST_KEY,
        pair_2nd_key: str = PAIR_2ND_KEY,
        index_key: str = INDEX_KEY,
    ):
        """

        Args:
            embeddings1: The first input embeddings
            embeddings2: The second input embeddings
            pair_1st_key: Key to put ``embeddings1`` into the batches
            pair_2nd_key: Key to put ``embeddings2`` into the batches
            index_key: Key to put samples' ids into the batches

        """
        assert embeddings1.shape == embeddings2.shape
        assert embeddings1.ndim >= 2

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key
        self.index_key = index_key

        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {self.pair_1st_key: self.embeddings1[idx], self.pair_2nd_key: self.embeddings2[idx], self.index_key: idx}

    def __len__(self) -> int:
        return len(self.embeddings1)


class ImagePairsDataset(IPairsDataset):
    """
    Dataset to iterate over pairs of images.

    """

    def __init__(
        self,
        paths1: List[Path],
        paths2: List[Path],
        bboxes1: Optional[TBBoxes] = None,
        bboxes2: Optional[TBBoxes] = None,
        transform: Optional[TTransforms] = None,
        f_imread: TImReader = imread_pillow,
        pair_1st_key: str = PAIR_1ST_KEY,
        pair_2nd_key: str = PAIR_2ND_KEY,
        index_key: str = INDEX_KEY,
        cache_size: Optional[int] = 0,
    ):
        """
        Args:
            paths1: Paths to the 1st input images
            paths2: Paths to the 2nd input images
            bboxes1: Should be either ``None`` or a sequence of bboxes.
                If an image has ``N`` boxes, duplicate its
                path ``N`` times and provide bounding box for each of them.
                If you want to get an embedding for the whole image, set bbox to ``None`` for
                this particular image path. The format is ``x1, y1, x2, y2``.
            bboxes2: The same as ``bboxes2``, but for the second inputs.
            transform: Augmentations for the images, set ``None`` to perform only normalisation and casting to tensor
            f_imread: Function to read the images
            pair_1st_key: Key to put the 1st images into the batches
            pair_2nd_key: Key to put the 2nd images into the batches
            index_key: Key to put samples' ids into the batches
            cache_size: Size of the dataset's cache

        """
        assert len(paths1) == len(paths2)

        if transform is None:
            transform = get_normalisation_torch()

        cache_size = cache_size // 2 if cache_size else None
        dataset_args = {"transform": transform, "f_imread": f_imread, "cache_size": cache_size}
        self.dataset1 = ListDataset(paths1, bboxes=bboxes1, **dataset_args)
        self.dataset2 = ListDataset(paths2, bboxes=bboxes2, **dataset_args)

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key
        self.index_key = index_key

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Dict[str, Any]]]:
        return {self.pair_1st_key: self.dataset1[idx], self.pair_2nd_key: self.dataset2[idx], self.index_key: idx}

    def __len__(self) -> int:
        return len(self.dataset1)


__all__ = ["EmbeddingPairsDataset", "ImagePairsDataset"]
