from typing import Dict

from oml.const import PAIR_1ST_KEY, PAIR_2ND_KEY
from oml.interfaces.datasets import IPairsDataset
from torch import Tensor


class EmbeddingsPairsDataset(IPairsDataset):
    """
    Dataset to iterate over pairs of embeddings.

    """

    def __init__(self,
                 embeddings1: Tensor,
                 embeddings2: Tensor,
                 pair_1st_key: str = PAIR_1ST_KEY,
                 pair_2nd_key: str = PAIR_2ND_KEY
                 ):
        assert embeddings1.shape == embeddings2.shape
        assert embeddings1.ndim >= 2

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key

        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            self.pair_1st_key: self.embeddings1[idx],
            self.pair_2nd_key: self.embeddings2[idx]
        }

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
        assert len(paths1) == len(paths2)

        dataset_args = {"bboxes": None, "transform": transform, "f_imread": f_imread, "cache_size": cache_size // 2}
        self.dataset1 = ListDataset(paths1, **dataset_args)
        self.dataset2 = ListDataset(paths2, **dataset_args)

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {self.pair_1st_key: self.dataset1[idx], self.pair_2nd_key: self.dataset2[idx]}

    def __len__(self) -> int:
        return len(self.dataset1)


__all__ = ["EmbeddingsPairsDataset", "ImagePairsDataset"]
