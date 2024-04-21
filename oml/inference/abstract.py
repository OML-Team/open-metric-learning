from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch import FloatTensor, Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from oml.datasets import PairDataset
from oml.ddp.patching import patch_dataloader_to_ddp
from oml.ddp.utils import get_world_size_safe, is_ddp, sync_dicts_ddp
from oml.interfaces.datasets import IBaseDataset, IIndexedDataset
from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import get_device, temporary_setting_model_mode, unique_by_ids


@torch.no_grad()
def _inference(
    model: nn.Module,
    apply_model: Callable[[nn.Module, Dict[str, Any]], Tensor],
    dataset: IIndexedDataset,
    num_workers: int,
    batch_size: int,
    verbose: bool,
    use_fp16: bool,
    accumulate_on_cpu: bool = True,
) -> Tensor:
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

    ids, outputs = unique_by_ids(ids=ids, data=outputs)

    assert len(outputs) == len(dataset), "Data was not collected correctly after DDP sync."
    assert list(range(len(dataset))) == ids, "Data was not collected correctly after DDP sync."

    return outputs


@torch.no_grad()
def inference(
    model: nn.Module,
    dataset: IBaseDataset,
    batch_size: int,
    num_workers: int = 0,
    verbose: bool = False,
    use_fp16: bool = False,
    accumulate_on_cpu: bool = True,
) -> Tensor:
    device = get_device(model)

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


def inference_cached(
    model: nn.Module,
    dataset: IBaseDataset,
    batch_size: int,
    num_workers: int = 0,
    use_fp16: bool = False,
    verbose: bool = True,
    accumulate_on_cpu: bool = True,
    cache_path: str = "inference_cache.pth",
) -> Tensor:
    if Path(cache_path).is_file():
        outputs = torch.load(cache_path, map_location="cpu")
        print(f"Model outputs have been loaded from {cache_path}.")
    else:
        outputs = inference(
            model=model,
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            use_fp16=use_fp16,
            verbose=verbose,
            accumulate_on_cpu=accumulate_on_cpu,
        )

        torch.save(outputs, cache_path)
        print(f"Model outputs have been saved to {cache_path}.")

    return outputs


def pairwise_inference(
    model: IPairwiseModel,
    base_dataset: IBaseDataset,
    pair_ids: List[Tuple[int, int]],
    num_workers: int,
    batch_size: int,
    verbose: bool = True,
    use_fp16: bool = False,
    accumulate_on_cpu: bool = True,
) -> Tensor:
    device = get_device(model)

    dataset = PairDataset(base_dataset=base_dataset, pair_ids=pair_ids)

    def _apply(
        model_: IPairwiseModel,
        batch_: Dict[str, Any],
    ) -> Tensor:
        pair1 = batch_[dataset.input_tensors_key_1].to(device)
        pair2 = batch_[dataset.input_tensors_key_2].to(device)
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


__all__ = ["inference", "pairwise_inference", "inference_cached"]
