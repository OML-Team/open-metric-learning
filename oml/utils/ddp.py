import itertools
import warnings
from typing import List, Dict, Any, Optional, Union, Iterator, Sequence

import numpy as np
import torch
from torch.distributed import all_gather_object, get_rank, is_initialized, logging, get_world_size
from torch.utils.data import Dataset, DistributedSampler, DataLoader, BatchSampler

TGenericSampler = Union[Iterator[int], Iterator[List[int]]]


class WarningDDP(UserWarning):
    def __init__(self, function_name: str):
        self.message = f"You use '{function_name}' function, but default process group has not been initialized. " \
                       "In this case for compatibility function doesn't modify your data. " \
                       "Please make sure to call init_process_group or ignore this warning if you want use this " \
                       "function without DDP"

    def __str__(self):
        return repr(self.message)


def merge_list_of_dicts(list_of_dicts: List[Dict[str, Any]], device: torch.device = 'cpu') -> Dict[str, Any]:
    available_types = (list, tuple, torch.Tensor, np.ndarray)

    for dict_sample in list_of_dicts:
        assert all(isinstance(v, available_types) for v in dict_sample.values()), \
            f"Only '{available_types}' are available for merging. " \
            f"Got types '{tuple((k, type(v)) for k, v in list_of_dicts[0].items())}'"
        assert set((k, type(v)) for k, v in list_of_dicts[0].items()) == \
               set((k, type(v)) for k, v in dict_sample.items()),\
               "Dictionaries in list have to have same keys with same types of values"

    output = {}

    for key in list_of_dicts[0].keys():
        if isinstance(list_of_dicts[0][key], (list, tuple)):
            output[key] = list(itertools.chain(*tuple(g[key] for g in list_of_dicts)))
        elif isinstance(list_of_dicts[0][key], torch.Tensor):
            output[key] = torch.cat([g[key].to(device) for g in list_of_dicts], dim=0)
        elif isinstance(list_of_dicts[0][key], np.ndarray):
            output[key] = np.concatenate(tuple(g[key] for g in list_of_dicts), axis=0)
        else:
            raise TypeError(f"Only '{available_types}' are available for merging. "
                            f"Got type '{type(list_of_dicts[0][key])}' for key '{key}'")

    return output


def sync_dicts_ddp(outputs_from_device: Dict[str, Any], world_size: int, device: Union[torch.device, str] = 'cpu') -> Dict[str, Any]:
    """
    Function allows you to combine and merge data stored in dict from all devices. You can place this function in
    your code and all devices upon reaching this function will wait for each other to synchronize and merging dicts.

    NOTE: Function under the hood pickles all object, convert bytes to tensor, then unpickle after syncing.
    With nccl (default) DDP backend intermediate tensors are stored on CUDA.
    """
    if world_size > 1 and is_initialized():
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        all_gather_object(gathered, outputs_from_device, group=torch.distributed.group.WORLD)
        return merge_list_of_dicts(gathered, device)
    else:
        if world_size > 1:
            warnings.warn(sync_dicts_ddp.__name__, WarningDDP)
        return outputs_from_device


class _Sampler2Dataset(Dataset):
    def __init__(self, sampler: TGenericSampler):
        self.sampler = sampler
        self.sampler_readed = None

    def __getitem__(self, item: int) -> Union[int, List[int]]:
        if self.sampler_readed is None:
            self.sampler_readed = list(self.sampler)

        return self.sampler_readed[item]

    def __len__(self) -> int:
        return len(self.sampler)


class DDPSamplerWrapper(DistributedSampler):
    def __init__(self,
                 sampler: TGenericSampler,
                 shuffle: bool = True,
                 drop_last: bool = False
                 ):
        super().__init__(dataset=_Sampler2Dataset(sampler), shuffle=shuffle, drop_last=drop_last)
        """
        Default DistributedSampler allows us to built sampler for dataset in DDP mode. Usually we can easily replace 
        default SequentialSampler (when shuffle=False) and RandomSampler (when shuffle=True) with DistributedSampler. 
        But for custom sampler we need extra wrapper. With this class we mimic any samplers to dataset and use indices
        of sampler splitted for each divices. Using this distributed indices we can prepare batch of data.
        
        When using DDP we should manage behaviour with last batch, because each device should have the same ammount of 
        data
        
        Args:
            sampler: sequential or batch sampler
            drop_last: When using DDP we should manage behaviour with last batch, because each device should have the 
            same ammount of data. If the sampler length is not evenly divisible by the number of devices, we must 
            duplicate part of the data (drop_last=False), or discard part of the data (drop_last=True).
            shuffle: allow shuffle indices of sampler
        
        Note: Wrapper also can be used with default SequentialSampler and RandomSampler, not only custom.
        """

        self.shift_per_epoch = 0
        self.sampler = sampler

    def _reload(self) -> None:
        """
        We need to reinstantiate wrapper in order to update available indices from sampler for new epoch
        """
        self.shift_per_epoch += 1
        DistributedSampler.__init__(self, dataset=_Sampler2Dataset(self.sampler), shuffle=self.shuffle, drop_last=self.drop_last)
        self.set_epoch(self.shift_per_epoch)

    def __iter__(self) -> TGenericSampler:
        self._reload()

        for sampler_idx in super().__iter__():
            output = self.dataset[sampler_idx]
            # if isinstance(output, list):
                # print(get_rank(), self.shift_per_epoch, sampler_idx, output)

            yield output


def patch_dataloader_to_ddp(loader: DataLoader, drop_last: bool, shuffle: bool) -> DataLoader:
    """
    Function inspects loader and modifies sampler for working in DDP mode. Behaviour with and without DDP may be
    slightly different due to drop_last option. For more details, see DDPSamplerWrapper docs.
    """
    if is_initialized():
        # If you don't spectify batch_sampler, PyTorch automatically creates default BatchSampler. In this case we
        # need convert to DDP only sampler (your custom sampler / default SequentialSampler or RandomSampler, which
        # PyTorch creates if sampler=None). We don't use `isinstance(...)` for `if` statement because we need exactly
        # class BatchSampler, ignoring any inheritance
        if type(loader.batch_sampler) is BatchSampler:
            ddp_sampler = DDPSamplerWrapper(sampler=loader.sampler,
                                            shuffle=shuffle,
                                            drop_last=drop_last)
            patched_loader = DataLoader(dataset=loader.dataset,
                                sampler=ddp_sampler,
                                batch_size=loader.batch_size,
                                num_workers=loader.num_workers)
            sampler_info = f"'{loader.sampler.__class__.__name__}' sampler"
        else:
            ddp_sampler = DDPSamplerWrapper(sampler=loader.batch_sampler,
                                            shuffle=shuffle,
                                            drop_last=drop_last)
            patched_loader = DataLoader(dataset=loader.dataset,
                                        batch_sampler=ddp_sampler,
                                        num_workers=loader.num_workers)
            sampler_info = f"'{loader.batch_sampler.__class__.__name__}' batch sampler"

        logging.info(f"DataLoader with {sampler_info} is updated to DDP mode "
                     f"with drop_last={drop_last} and shuffle={shuffle} parameters")
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
