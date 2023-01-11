from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from oml.const import PAIR_1ST_KEY, PAIR_2ND_KEY
from oml.datasets.list_dataset import ListDataset
from oml.interfaces.models import IPairwiseDistanceModel
from oml.transforms.images.torchvision.transforms import get_normalisation_torch
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader, imread_cv2


class VectorsPairsDataset(Dataset):
    """
    Dataset to iterate over pairs of embeddings.

    """

    def __init__(self, x1: Tensor, x2: Tensor, pair_1st_key: str = PAIR_1ST_KEY, pair_2nd_key: str = PAIR_2ND_KEY):
        assert x1.shape == x2.shape
        assert x1.ndim >= 2

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key

        self.x1 = x1
        self.x2 = x2

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {self.pair_1st_key: self.x1[idx], self.pair_2nd_key: self.x2[idx]}

    def __len__(self) -> int:
        return len(self.x1)


class ImagePairsDataset(Dataset):
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


def images_pairwise_inference(
    model: IPairwiseDistanceModel,
    paths1: List[Path],
    paths2: List[Path],
    transform: TTransforms,
    f_imread: Optional[TImReader] = None,
    num_workers: int = 0,
    batch_size: int = 512,
    verbose: bool = False,
) -> Tensor:
    if f_imread is None:
        f_imread = get_im_reader_for_transforms(transform)
    dataset = ImagePairsDataset(paths1=paths1, paths2=paths2, transform=transform, f_imread=f_imread)
    output = pairwise_inference(
        model=model, dataset=dataset, num_workers=num_workers, batch_size=batch_size, verbose=verbose
    )
    return output


# todo: we need interfaces for model and dataset below
def vectors_pairwise_inference(
    model: nn.Module, x1: Tensor, x2: Tensor, num_workers: int = 0, batch_size: int = 512, verbose: bool = False
) -> Tensor:
    dataset = VectorsPairsDataset(x1=x1, x2=x2)
    output = pairwise_inference(
        model=model, dataset=dataset, num_workers=num_workers, batch_size=batch_size, verbose=verbose
    )
    return output


# todo: we need interfaces for model and dataset below
def pairwise_inference(
    model: nn.Module, dataset: Dataset, num_workers: int = 0, batch_size: int = 512, verbose: bool = False
) -> Tensor:
    prev_mode = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    device = next(model.parameters()).device

    loader = tqdm(loader) if verbose else loader

    outputs = []
    with torch.no_grad():
        for batch in loader:
            x1 = batch[dataset.pair_1st_key].to(device)
            x2 = batch[dataset.pair_2nd_key].to(device)
            outputs.append(model(x1=x1, x2=x2))

    model.train(prev_mode)
    return torch.cat(outputs).detach().cpu()


__all__ = [
    "ImagePairsDataset",
    "VectorsPairsDataset",
    "images_pairwise_inference",
    "vectors_pairwise_inference",
    "pairwise_inference",
]
