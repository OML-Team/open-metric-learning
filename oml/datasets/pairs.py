from pathlib import Path
from typing import Dict, List

from torch import Tensor

from oml.const import PAIR_1ST_KEY, PAIR_2ND_KEY
from oml.datasets.list_dataset import ListDataset
from oml.interfaces.datasets import IPairsDataset
from oml.transforms.images.torchvision.transforms import get_normalisation_torch
from oml.transforms.images.utils import TTransforms
from oml.utils.images.images import TImReader, imread_cv2


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
    ):
        """

        Args:
            embeddings1: The first input embeddings
            embeddings2: The second input embeddings
            pair_1st_key: Key to put ``embeddings1`` into the batches
            pair_2nd_key: Key to put ``embeddings2`` into the batches

        """
        assert embeddings1.shape == embeddings2.shape
        assert embeddings1.ndim >= 2

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key

        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {self.pair_1st_key: self.embeddings1[idx], self.pair_2nd_key: self.embeddings2[idx]}

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
        transform: TTransforms = get_normalisation_torch(),
        f_imread: TImReader = imread_cv2,
        pair_1st_key: str = PAIR_1ST_KEY,
        pair_2nd_key: str = PAIR_2ND_KEY,
        cache_size: int = 100_000,
    ):
        """
        Args:
            paths1: Paths to the 1st input images
            paths2: Paths to the 2nd input images
            transform: Augmentations for the images, set ``None`` to perform only normalisation and casting to tensor
            f_imread: Function to read the images
            pair_1st_key: Key to put the 1st images into the batches
            pair_2nd_key: Key to put the 2nd images into the batches
            cache_size: Size of the dataset's cache

        """
        assert len(paths1) == len(paths2)

        dataset_args = {"bboxes": None, "transform": transform, "f_imread": f_imread, "cache_size": cache_size // 2}
        self.dataset1 = ListDataset(paths1, **dataset_args)
        self.dataset2 = ListDataset(paths2, **dataset_args)

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # todo: add support of bounding boxes
        return {self.pair_1st_key: self.dataset1[idx], self.pair_2nd_key: self.dataset2[idx]}

    def __len__(self) -> int:
        return len(self.dataset1)


__all__ = ["EmbeddingPairsDataset", "ImagePairsDataset"]
