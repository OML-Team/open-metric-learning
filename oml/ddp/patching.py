import logging
import warnings
from typing import List, Sequence, Union

from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
)

from oml.ddp.utils import WarningDDP, is_ddp
from oml.interfaces.samplers import IBatchSampler

TAllSamplers = Union[BatchSampler, Sampler, IBatchSampler]


class _Sampler2Dataset(Dataset):
    def __init__(self, sampler: TAllSamplers):
        # We read sampler in __getitem__ due to the seed between calling of __init__ and __getitem__ can be changed
        self.sampler_read = None
        self.sampler = sampler

    def __getitem__(self, item: int) -> Union[int, List[int]]:
        if self.sampler_read is None:
            self.sampler_read = list(self.sampler)  # type: ignore

        return self.sampler_read[item]  # type: ignore

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
        Default DistributedSampler allows us to build a sampler for a dataset in DDP mode. Usually we can easily replace
        default SequentialSampler (when DataLoader(shuffle=False, ...)) and RandomSampler
        (when DataLoader(shuffle=True, ...)) with DistributedSampler. But for the custom sampler, we need an extra
        wrapper.
        With this class, we mimic any type of samplers to a dataset and use indices of sampler splitted for each device.
        Using these distributed indices we can prepare a batch of data.

        Args:
            sampler: sequential or batch sampler
            pad_data_to_num_gpus: When using DDP we should manage behavior with the last batch, because each device
                should have the same amount of data. If the sampler length is not evenly divisible by the number of
                devices, we must duplicate part of the data (pad_data_to_num_gpus=True), or discard part of the
                data (pad_data_to_num_gpus=False).
            shuffle_samples_between_gpus: shuffle available indices between feeding to GPU. Note, that shuffle
                inside GPU after feeding will be used according to behavior of sampler.
        Note: Wrapper also can be used with default SequentialSampler and RandomSampler, not only custom.
        """

        self.seed_shift_per_epoch = 0
        self.sampler = sampler

    def _reload(self) -> None:
        """
        We need to reinstantiate wrapper in order to update available indices from sampler for new epoch
        """
        if self.seed_shift_per_epoch > 0:
            super().__init__(dataset=_Sampler2Dataset(self.sampler), shuffle=self.shuffle, drop_last=self.drop_last)
            self.set_epoch(self.seed_shift_per_epoch)
        self.seed_shift_per_epoch += 1

    def __iter__(self) -> TAllSamplers:
        self._reload()

        for sampler_idx in super().__iter__():
            yield self.dataset[sampler_idx]


def patch_dataloader_to_ddp(loader: DataLoader) -> DataLoader:
    """
    Function inspects loader and modifies sampler for working in DDP mode.
    Note:
        We ALWAYS use the padding of samples (number of batches or number of samples per epoch) in order to use the
        same amount of data for each device in DDP, so behavior with and without DDP may be slightly
        different (e.g. metrics).
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
            ddp_sampler = DDPSamplerWrapper(
                sampler=loader.sampler, shuffle_samples_between_gpus=False, pad_data_to_num_gpus=True
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


__all__ = [
    "DDPSamplerWrapper",
    "patch_dataloader_to_ddp",
    "check_loaders_is_patched",
]
