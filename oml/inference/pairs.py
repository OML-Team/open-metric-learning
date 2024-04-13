from typing import Any, Dict, List, Tuple

from torch import FloatTensor, Tensor

from oml.datasets.base import BaseDataset
from oml.datasets.pairs import PairsDataset
from oml.inference.abstract import _inference
from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import get_device


def pairwise_inference(
    model: IPairwiseModel,
    base_dataset: BaseDataset,
    pair_ids: List[Tuple[int, int]],
    num_workers: int,
    batch_size: int,
    verbose: bool = True,
    use_fp16: bool = False,
    accumulate_on_cpu: bool = True,
) -> FloatTensor:
    device = get_device(model)

    dataset = PairsDataset(base_dataset=base_dataset, pair_ids=pair_ids)

    def _apply(
        model_: IPairwiseModel,
        batch_: Dict[str, Any],
    ) -> Tensor:
        pair1 = batch_[dataset.pair_1st_key].to(device)
        pair2 = batch_[dataset.pair_2nd_key].to(device)
        return model_.predict(pair1, pair2)

    output = _inference(
        model=model,
        apply_model=_apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
        use_fp16=use_fp16,
        accumulate_on_cpu=accumulate_on_cpu,
    )

    return output


__all__ = ["pairwise_inference"]
