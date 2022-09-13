import itertools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.distributed import all_gather_object, get_world_size, logging
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    RandomSampler,
    Sampler,
)

from oml.interfaces.samplers import IBatchSampler

TAllSamplers = Union[BatchSampler, Sampler, IBatchSampler]


class WarningDDP(UserWarning):
    def __init__(self, function_name: str):
        self.message = (
            f"You use '{function_name}' function, but default process group has not been initialized. "
            "In this case for compatibility function doesn't modify your data. "
            "Please make sure to call init_process_group or ignore this warning if you want use this "
            "function without DDP"
        )

    def __str__(self) -> str:
        return repr(self.message)


def merge_list_of_dicts(list_of_dicts: List[Dict[str, Any]], device: torch.device = "cpu") -> Dict[str, Any]:
    available_types = (list, tuple, torch.Tensor, np.ndarray)

    for dict_sample in list_of_dicts:
        assert all(isinstance(v, available_types) for v in dict_sample.values()), (
            f"Only '{available_types}' are available for merging. "
            f"Got types '{tuple((k, type(v)) for k, v in list_of_dicts[0].items())}'"
        )
        assert set((k, type(v)) for k, v in list_of_dicts[0].items()) == set(
            (k, type(v)) for k, v in dict_sample.items()
        ), "Dictionaries in list have to have same keys with same types of values"

    output = {}

    for key in list_of_dicts[0].keys():
        if isinstance(list_of_dicts[0][key], (list, tuple)):
            output[key] = list(itertools.chain(*tuple(g[key] for g in list_of_dicts)))
        elif isinstance(list_of_dicts[0][key], torch.Tensor):
            output[key] = torch.cat([g[key].to(device) for g in list_of_dicts], dim=0)
        elif isinstance(list_of_dicts[0][key], np.ndarray):
            output[key] = np.concatenate(tuple(g[key] for g in list_of_dicts), axis=0)
        else:
            raise TypeError(
                f"Only '{available_types}' are available for merging. "
                f"Got type '{type(list_of_dicts[0][key])}' for key '{key}'"
            )

    return output


def sync_dicts_ddp(
    outputs_from_device: Dict[str, Any], world_size: int, device: Union[torch.device, str] = "cpu"
) -> Dict[str, Any]:
    """
    Function allows you to combine and merge data stored in dict from all devices. You can place this function in
    your code and all devices upon reaching this function will wait for each other to synchronize and merging dicts.
    NOTE: Function under the hood pickles all object, convert bytes to tensor, then unpickle after syncing.
    With nccl (default) DDP backend intermediate tensors are stored on CUDA.
    """
    if world_size >= 1 and is_ddp():
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        all_gather_object(gathered, outputs_from_device, group=torch.distributed.group.WORLD)
        return merge_list_of_dicts(gathered, device)
    else:
        if world_size > 1:
            warnings.warn(sync_dicts_ddp.__name__, WarningDDP)
        return outputs_from_device


class _Sampler2Dataset(Dataset):
    def __init__(self, sampler: TAllSamplers):
        self.sampler = sampler
        self.sampler_readed = None

    def __getitem__(self, item: int) -> Union[int, List[int]]:
        if self.sampler_readed is None:
            self.sampler_readed = list(self.sampler)  # type: ignore

        return self.sampler_readed[item]  # type: ignore

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore


class DDPSamplerWrapper(DistributedSampler):
    def __init__(
        self, sampler: TAllSamplers, shuffle_samples_between_gpus: bool = True, pad_data_to_num_gpus: bool = True
    ):
        super().__init__(
            dataset=_Sampler2Dataset(sampler), shuffle=shuffle_samples_between_gpus, drop_last=not pad_data_to_num_gpus
        )
        """
        Default DistributedSampler allows us to built sampler for dataset in DDP mode. Usually we can easily replace
        default SequentialSampler (when DataLoader(shuffle=False, ...)) and RandomSampler
        (when DataLoader(shuffle=True, ...)) with DistributedSampler. But for custom sampler we need extra wrapper.
        With this class we mimic any samplers to dataset and use indices of sampler splitted for each divices.
        Using this distributed indices we can prepare batch of data.
        When using DDP we should manage behaviour with last batch, because each device should have the same ammount of
        data
        Args:
            sampler: sequential or batch sampler
            pad_data_to_num_gpus: When using DDP we should manage behaviour with last batch, because each device should
                have the same ammount of data. If the sampler length is not evenly divisible by the number of devices,
                we must duplicate part of the data (pad_data_to_num_gpus=True), or discard part of the
                data (pad_data_to_num_gpus=False).
            shuffle_samples_between_gpus: shuffle available indices between feeding to gpu. Note, that shuffle
            inside gpu after feeding will be used according behaviour of sampler.
        Note: Wrapper also can be used with default SequentialSampler and RandomSampler, not only custom.
        """

        self.shift_per_epoch = 0
        self.sampler = sampler

    def _reload(self) -> None:
        """
        We need to reinstantiate wrapper in order to update available indices from sampler for new epoch
        """
        self.shift_per_epoch += 1
        super().__init__(dataset=_Sampler2Dataset(self.sampler), shuffle=self.shuffle, drop_last=self.drop_last)
        self.set_epoch(self.shift_per_epoch)

    def __iter__(self) -> TAllSamplers:
        self._reload()

        for sampler_idx in super().__iter__():
            yield self.dataset[sampler_idx]


def patch_dataloader_to_ddp(loader: DataLoader) -> DataLoader:
    """
    Function inspects loader and modifies sampler for working in DDP mode.
    Note:
        We ALWAYS use padding of samples (number of batches or number of samples per epoch) in order to use same amount
        of data for each device in DDP, so behaviour with and without DDP may be slightly different (e.g. metrics).
    """
    if is_ddp():
        kwargs_loader = {
            "collate_fn": loader.collate_fn,
            "persistent_workers": loader.persistent_workers,
            "pin_memory": loader.pin_memory,
            "worker_init_fn": loader.worker_init_fn,
            "prefetch_factor": loader.prefetch_factor,
            "multiprocessing_context": loader.multiprocessing_context,
            "pin_memory_device": loader.pin_memory_device,
            "num_workers": loader.num_workers,
        }

        # If you don't spectify batch_sampler, PyTorch automatically creates default BatchSampler. In this case we
        # need convert to DDP only sampler (your custom sampler / default SequentialSampler or RandomSampler, which
        # PyTorch creates if sampler=None). We don't use `isinstance(...)` for `if` statement because we need exactly
        # class BatchSampler, ignoring any inheritance
        if type(loader.batch_sampler) is BatchSampler:
            shuffle = True if isinstance(loader.sampler, RandomSampler) else False
            ddp_sampler = DDPSamplerWrapper(
                sampler=loader.sampler, shuffle_samples_between_gpus=shuffle, pad_data_to_num_gpus=True
            )
            patched_loader = DataLoader(
                dataset=loader.dataset,
                sampler=ddp_sampler,
                batch_size=loader.batch_size,
                drop_last=loader.drop_last,
                **kwargs_loader,
            )
            sampler_info = f"'{loader.sampler.__class__.__name__}' sampler"
        else:
            ddp_sampler = DDPSamplerWrapper(
                sampler=loader.batch_sampler, shuffle_samples_between_gpus=True, pad_data_to_num_gpus=True
            )
            patched_loader = DataLoader(dataset=loader.dataset, batch_sampler=ddp_sampler, **kwargs_loader)
            sampler_info = f"'{loader.batch_sampler.__class__.__name__}' batch sampler"

        logging.info(f"DataLoader with {sampler_info} is updated to DDP mode")
        return patched_loader
    else:
        warnings.warn(patch_dataloader_to_ddp.__name__, WarningDDP)
        return loader


def check_loaders_is_patched(loaders: Union[DataLoader, Sequence[DataLoader]]) -> bool:
    loaders = [loaders] if isinstance(loaders, DataLoader) else loaders

    for loader in loaders:
        if not any(isinstance(sampler, DDPSamplerWrapper) for sampler in [loader.batch_sampler, loader.sampler]):
            return False

    return True


def is_ddp() -> bool:
    try:
        get_world_size()
        return True
    except RuntimeError:
        return False


__all__ = [
    "is_ddp",
    "patch_dataloader_to_ddp",
    "check_loaders_is_patched",
    "sync_dicts_ddp",
    "merge_list_of_dicts",
    "DDPSamplerWrapper",
]
