from typing import Any, Callable, Dict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from oml.ddp.patching import patch_dataloader_to_ddp
from oml.ddp.utils import get_world_size_safe, is_ddp, sync_dicts_ddp
from oml.utils.misc_torch import drop_duplicates_by_ids, temporary_setting_model_mode


@torch.no_grad()
def _inference(
    model: nn.Module,
    apply_model: Callable[[nn.Module, Dict[str, Any]], Tensor],
    dataset: Dataset,  # type: ignore
    num_workers: int,
    batch_size: int,
    verbose: bool,
) -> Tensor:
    assert hasattr(dataset, "index_key"), "We expect that your dataset returns samples ids in __getitem__ method"

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if is_ddp():
        loader = patch_dataloader_to_ddp(loader)
    else:
        loader = tqdm(loader) if verbose else loader

    outputs_list = []
    ids = []
    with temporary_setting_model_mode(model, set_train=False):
        for batch in loader:
            out = apply_model(model, batch)
            outputs_list.append(out)
            ids.extend(batch[dataset.index_key].long().tolist())

    outputs = torch.cat(outputs_list).detach()

    data_to_sync = {"outputs": outputs, "ids": ids}
    data_synced = sync_dicts_ddp(data_to_sync, world_size=get_world_size_safe())
    outputs, ids = data_synced["outputs"], data_synced["ids"]

    ids, outputs = drop_duplicates_by_ids(ids=ids, data=outputs, sort=True)

    assert len(outputs) == len(dataset), "Data was not collected correctly after DDP sync."
    assert list(range(len(dataset))) == ids, "Data was not collected correctly after DDP sync."

    return outputs


__all__ = ["_inference"]
