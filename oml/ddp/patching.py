import inspect
import logging
import warnings
from typing import Any, Dict, List, Sequence, Union

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
    """
    This is a wrapper to allow using custom sampler in DDP mode.

    Default `DistributedSampler` allows us to build a sampler for a dataset in DDP mode.
    Usually we can easily replace default `SequentialSampler` [when ``DataLoader(shuffle=False, ...)``] and
    `RandomSampler` [when ``DataLoader(shuffle=True, ...)``] with `DistributedSampler`. But for the custom sampler,
    we need an extra wrapper.

    Thus, this wrapper distributes indices produced by sampler among several devices for further usage.

    """

    def __init__(
        self, sampler: TAllSamplers, shuffle_samples_between_gpus: bool = True, pad_data_to_num_gpus: bool = True
    ):
        """
        Args:
            sampler: Sequential or batch sampler
            pad_data_to_num_gpus: When using DDP we should manage behavior with the last batch, because each device
                should have the same amount of data. If the sampler length is not evenly divisible by the number of
                devices, we must duplicate part of the data (``pad_data_to_num_gpus=True``), or discard part of the
                data (``pad_data_to_num_gpus=False``).
            shuffle_samples_between_gpus: shuffle available indices before feeding them to GPU. Note, that shuffle
                inside GPU after the feeding will be used according to behavior of the sampler.

        Note: Wrapper can be used with both the default `SequentialSampler` or `RandomSampler` from `PyTorch` and
        with some custom sampler.

        """
        super().__init__(
            dataset=_Sampler2Dataset(sampler), shuffle=shuffle_samples_between_gpus, drop_last=not pad_data_to_num_gpus
        )
        self.seed_shift_per_epoch = 0
        self.sampler = sampler

    def _reload(self) -> None:
        """
        We need to re-instantiate the wrapper in order to update the available indices for the new epoch.
        We don't perform this step on the epoch 0, because we want to be comparable with no DDP setup there.

        """
        if self.seed_shift_per_epoch > 0:
            super().__init__(dataset=_Sampler2Dataset(self.sampler), shuffle=self.shuffle, drop_last=self.drop_last)
            self.set_epoch(self.seed_shift_per_epoch)
        self.seed_shift_per_epoch += 1

    def __iter__(self) -> TAllSamplers:
        self._reload()

        for sampler_idx in super().__iter__():
            yield self.dataset[sampler_idx]


def extract_loader_parameters(loader: DataLoader, ignore_data_related_parameters: bool = False) -> Dict[str, Any]:
    """
    The function extracts parameters from dataloader, such as `collate_fn`, `num_workers`, etc, and automatically
    handles some new parameters, e.g. `prefetch_factor`.

    Args:
        loader: loader from which parameters are extracted
        ignore_data_related_parameters: The flag allows you to ignore parameters, related to data, batch content,
        and samplers.

    """

    ignore_fields = ["self"]

    if ignore_data_related_parameters:
        ignore_fields.extend(["sampler", "batch_sampler", "drop_last", "shuffle", "batch_size", "dataset"])

    extracted = {}

    signature = inspect.signature(DataLoader.__init__)
    for parameter in signature.parameters:
        if parameter not in ignore_fields:
            if hasattr(loader, parameter):
                extracted[parameter] = getattr(loader, parameter)

    assert len(extracted)

    return extracted


def patch_dataloader_to_ddp(loader: DataLoader) -> DataLoader:
    """
    Function inspects loader and modifies sampler for working in DDP mode.

    Note:
        We ALWAYS use the padding of samples (in terms of the number of batches or number of samples per epoch) in
        order to use the same amount of data for each device in DDP. Thus, the behavior with and without DDP may be
        slightly different (e.g. metrics values).

    """
    if is_ddp():
        kwargs_loader = extract_loader_parameters(loader, ignore_data_related_parameters=True)

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
                sampler=loader.batch_sampler, shuffle_samples_between_gpus=False, pad_data_to_num_gpus=True
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
    "extract_loader_parameters",
    "check_loaders_is_patched",
]
