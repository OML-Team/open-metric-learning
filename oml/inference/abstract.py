from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import FloatTensor, Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from oml.datasets import PairsDataset
from oml.ddp.patching import patch_dataloader_to_ddp
from oml.ddp.utils import get_world_size_safe, is_ddp, sync_dicts_ddp
from oml.interfaces.datasets import IBaseDataset
from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import (
    drop_duplicates_by_ids,
    get_device,
    temporary_setting_model_mode,
)


@torch.no_grad()
def _inference(
    model: nn.Module,
    apply_model: Callable[[nn.Module, Dict[str, Any]], FloatTensor],
    dataset: Dataset,  # type: ignore
    num_workers: int,
    batch_size: int,
    verbose: bool,
    use_fp16: bool,
    accumulate_on_cpu: bool = True,
) -> FloatTensor:
    # todo: rework hasattr later
    assert hasattr(dataset, "index_key"), "We expect that your dataset returns samples ids in __getitem__ method"

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if is_ddp():
        loader = patch_dataloader_to_ddp(loader)

    if verbose:
        loader = tqdm(loader, desc=str(get_device(model)))

    outputs_list = []
    ids = []

    with torch.autocast(device_type="cuda", dtype=torch.float16 if use_fp16 else torch.float32):
        with temporary_setting_model_mode(model, set_train=False):
            for batch in loader:
                out = apply_model(model, batch)
                if accumulate_on_cpu:
                    out = out.cpu()
                outputs_list.append(out)
                ids.extend(batch[dataset.index_key].long().tolist())

    outputs = torch.cat(outputs_list).detach()

    data_to_sync = {"outputs": outputs, "ids": ids}
    data_synced = sync_dicts_ddp(data_to_sync, world_size=get_world_size_safe())
    outputs, ids = data_synced["outputs"], data_synced["ids"]

    ids, outputs = drop_duplicates_by_ids(ids=ids, data=outputs, sort=True)

    assert len(outputs) == len(dataset), "Data was not collected correctly after DDP sync."
    assert list(range(len(dataset))) == ids, "Data was not collected correctly after DDP sync."

    return outputs.float()


@torch.no_grad()
def inference(
    model: nn.Module,
    dataset: IBaseDataset,
    batch_size: int,
    num_workers: int = 0,
    verbose: bool = False,
    use_fp16: bool = False,
    accumulate_on_cpu: bool = True,
) -> FloatTensor:
    device = get_device(model)

    # Inference on IBaseDataset

    def apply(model_: nn.Module, batch_: Dict[str, Any]) -> FloatTensor:
        return model_(batch_[dataset.input_tensors_key].to(device))

    return _inference(
        model=model,
        apply_model=apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
        use_fp16=use_fp16,
        accumulate_on_cpu=accumulate_on_cpu,
    )


def pairwise_inference(
    model: IPairwiseModel,
    base_dataset: IBaseDataset,
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


def inference_cached(
    dataset: IBaseDataset,
    extractor: nn.Module,
    output_cache_path: str = "inference_cache.pth",
    num_workers: int = 0,
    batch_size: int = 128,
    use_fp16: bool = False,
) -> FloatTensor:
    if Path(output_cache_path).is_file():
        outputs = torch.load(output_cache_path, map_location="cpu")
        print(f"Model outputs have been loaded from {output_cache_path}.")
    else:
        outputs = inference(
            model=extractor,
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            use_fp16=use_fp16,
            verbose=True,
            accumulate_on_cpu=True,
        )

        torch.save(outputs, output_cache_path)
        print(f"Model outputs have been saved to {output_cache_path}.")

    return outputs


__all__ = ["inference", "pairwise_inference", "inference_cached"]
