from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from oml.datasets.list_dataset import ListDataset
from oml.inference.abstract import _inference
from oml.transforms.images.utils import TTransforms
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
    use_fp16: bool = False,
) -> Tensor:
    dataset = ListDataset(paths, bboxes=None, transform=transform, f_imread=f_imread, cache_size=0)
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
        use_fp16=use_fp16,
    )

    return outputs


__all__ = ["inference_on_images"]
