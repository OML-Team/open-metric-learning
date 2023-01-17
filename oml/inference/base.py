from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from oml.datasets.list_dataset import ListDataset
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader


# todo: use lightning for half precision and DDP
def inference_on_images(
    model: nn.Module,
    paths: List[Path],
    transform: TTransforms,
    num_workers: int,
    batch_size: int,
    verbose: bool,
    f_imread: Optional[TImReader] = None,
) -> Tensor:
    if f_imread is None:
        f_imread = get_im_reader_for_transforms(transform)

    prev_mode = model.training
    model.eval()

    dataset = ListDataset(paths, bboxes=None, transform=transform, f_imread=f_imread)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    device = next(model.parameters()).device

    loader = tqdm(loader) if verbose else loader

    outputs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            outputs.append(model(batch.to(device)))

    model.train(prev_mode)
    return torch.cat(outputs).detach().cpu()


__all__ = ["inference_on_images"]
