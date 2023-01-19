from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from oml.datasets.pairs import EmbeddingPairsDataset, ImagePairsDataset
from oml.interfaces.datasets import IPairsDataset
from oml.interfaces.models import IPairwiseModel
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader
from oml.utils.misc_torch import get_device, temporary_setting_model_mode


# todo: use lightning here to work with ddp and half precision
# fmt: off
@torch.no_grad()
def pairwise_inference(
        model: IPairwiseModel,
        dataset: IPairsDataset,
        num_workers: int,
        batch_size: int,
        verbose: bool
) -> Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    loader = tqdm(loader) if verbose else loader
    device = get_device(model)
    outputs_list = []

    with temporary_setting_model_mode(model, set_train=False):
        for batch in loader:
            x1 = batch[dataset.pair_1st_key].to(device)
            x2 = batch[dataset.pair_2nd_key].to(device)
            outputs_list.append(model(x1=x1, x2=x2))

    outputs = torch.cat(outputs_list).detach().cpu()
    return outputs


def pairwise_inference_on_images(
        model: IPairwiseModel,
        paths1: List[Path],
        paths2: List[Path],
        transform: TTransforms,
        num_workers: int,
        batch_size: int,
        verbose: bool = True,
        f_imread: Optional[TImReader] = None,
) -> Tensor:
    if f_imread is None:
        f_imread = get_im_reader_for_transforms(transform)
    dataset = ImagePairsDataset(paths1=paths1, paths2=paths2, transform=transform, f_imread=f_imread)
    output = pairwise_inference(
        model=model, dataset=dataset, num_workers=num_workers, batch_size=batch_size, verbose=verbose
    )
    return output


def pairwise_inference_on_embeddings(
        model: IPairwiseModel,
        embeddings1: Tensor,
        embeddings2: Tensor,
        num_workers: int,
        batch_size: int,
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
