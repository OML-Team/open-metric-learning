from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from oml.datasets.list_dataset import ListDataset
from oml.inference.abstract import _inference
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader
from oml.utils.misc_torch import get_device


@torch.no_grad()
def inference_on_images(
    model: nn.Module,
    paths: List[Path],
    transform: TTransforms,
    num_workers: int,
    batch_size: int,
    verbose: bool = False,
    f_imread: Optional[TImReader] = None,
) -> Tensor:
    if f_imread is None:
        f_imread = get_im_reader_for_transforms(transform)

    dataset = ListDataset(paths, bboxes=None, transform=transform, f_imread=f_imread)
    device = get_device(model)

    def _apply(model_: nn.Module, batch_: Dict[str, Any]) -> Tensor:
        return model_(batch_[dataset.input_tensors_key].to(device))

    outputs = _inference(
        model=model,
        apply_model=_apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
    )

    return outputs


__all__ = ["inference_on_images"]
