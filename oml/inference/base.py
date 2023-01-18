from pathlib import Path
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from oml.datasets.list_dataset import ListDataset
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader
from oml.utils.misc_torch import get_device, temporary_setting_model_mode


# todo: use lightning for half precision and DDP
@torch.no_grad()
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

    dataset = ListDataset(paths, bboxes=None, transform=transform, f_imread=f_imread)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    loader = tqdm(loader) if verbose else loader
    device = get_device(model)
    outputs_list = []

    with temporary_setting_model_mode(model, set_train=False):
        for batch in loader:
            outputs_list.append(model(batch.to(device)))

    outputs = torch.cat(outputs_list).detach().cpu()
    return outputs


__all__ = ["inference_on_images"]
