from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from oml.datasets.pairs import EmbeddingPairsDataset, ImagePairsDataset
from oml.interfaces.datasets import IPairsDataset
from oml.interfaces.models import IPairwiseDistanceModel
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader


def pairwise_inference(
    model: IPairwiseDistanceModel, dataset: IPairsDataset, num_workers: int, batch_size: int, verbose: bool
) -> Tensor:
    prev_mode = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    device = next(model.parameters()).device

    loader = tqdm(loader) if verbose else loader

    outputs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x1 = batch[dataset.pair_1st_key].to(device)
            x2 = batch[dataset.pair_2nd_key].to(device)
            outputs.append(model(x1=x1, x2=x2))

    model.train(prev_mode)
    return torch.cat(outputs).detach().cpu()


def pairwise_inference_on_images(
    model: IPairwiseDistanceModel,
    paths1: List[Path],
    paths2: List[Path],
    transform: TTransforms,
    f_imread: Optional[TImReader] = None,
    num_workers: int = 20,
    batch_size: int = 128,
    verbose: bool = False,
) -> Tensor:
    if f_imread is None:
        f_imread = get_im_reader_for_transforms(transform)
    dataset = ImagePairsDataset(paths1=paths1, paths2=paths2, transform=transform, f_imread=f_imread)
    output = pairwise_inference(
        model=model, dataset=dataset, num_workers=num_workers, batch_size=batch_size, verbose=verbose
    )
    return output


def pairwise_inference_on_embeddings(
    model: IPairwiseDistanceModel,
    embeddings1: Tensor,
    embeddings2: Tensor,
    num_workers: int = 0,
    batch_size: int = 512,
    verbose: bool = False,
) -> Tensor:
    dataset = EmbeddingPairsDataset(embeddings1=embeddings1, embeddings2=embeddings2)
    output = pairwise_inference(
        model=model, dataset=dataset, num_workers=num_workers, batch_size=batch_size, verbose=verbose
    )
    return output


__all__ = [
    "pairwise_inference_on_images",
    "pairwise_inference_on_embeddings",
    "pairwise_inference",
]
