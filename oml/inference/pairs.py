from pathlib import Path
from typing import Any, Dict, List, Optional

from torch import Tensor

from oml.datasets.pairs import EmbeddingPairsDataset, ImagePairsDataset
from oml.inference.abstract import _inference
from oml.interfaces.models import IPairwiseModel
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader
from oml.utils.misc_torch import get_device


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
    device = get_device(model)

    dataset = ImagePairsDataset(
        paths1=paths1,
        paths2=paths2,
        transform=transform,
        f_imread=get_im_reader_for_transforms(transform) if f_imread is None else f_imread,
    )

    def _apply(
        model_: IPairwiseModel,
        batch_: Dict[str, Any],
    ) -> Tensor:
        pair1 = batch_[dataset.pair_1st_key][dataset.dataset1.input_tensors_key].to(device)
        pair2 = batch_[dataset.pair_2nd_key][dataset.dataset2.input_tensors_key].to(device)
        return model_(pair1, pair2)

    output = _inference(
        model=model,
        apply_model=_apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
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
    device = get_device(model)

    dataset = EmbeddingPairsDataset(embeddings1=embeddings1, embeddings2=embeddings2)

    def _apply(
        model_: IPairwiseModel,
        batch_: Dict[str, Any],
    ) -> Tensor:
        pair1 = batch_[dataset.pair_1st_key].to(device)
        pair2 = batch_[dataset.pair_2nd_key].to(device)
        return model_(pair1, pair2)

    output = _inference(
        model=model,
        apply_model=_apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
    )

    return output


__all__ = [
    "pairwise_inference_on_images",
    "pairwise_inference_on_embeddings",
]
