from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, EPOCH_OUTPUT
from torch import nn
from torch.distributed import get_rank, get_world_size
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DistributedSampler, DataLoader, Dataset, BatchSampler, Sampler

from oml.interfaces.models import IExtractor


class Sampler2Dataset(Dataset):
    def __init__(self, batch_sampler: BatchSampler):
        self.sampler = batch_sampler
        self.sampler_readed = None

    def __getitem__(self, item: int) -> List[int]:
        if self.sampler_readed is None:
            self.sampler_readed = list(self.sampler)

        return self.sampler_readed[item]

    def __len__(self):
        return len(self.sampler)


class DDPSamplerWrapper(DistributedSampler):
    def __init__(self,
                 sampler: Sampler2Dataset,
                 shuffle: bool = True,
                 drop_last: bool = False
                 ):
        super().__init__(dataset=Sampler2Dataset(sampler), shuffle=shuffle, drop_last=drop_last)
        self.epoch_shift = 0

    def __iter__(self):
        for sampler_idx in super().__iter__():
            print(get_rank(), self.dataset[sampler_idx])
            output = self.dataset[sampler_idx]
            yield output
        self.epoch_shift += 1
        self.set_epoch(self.epoch_shift)


class ModuleDDP(pl.LightningModule):
    def __init__(self,
                 loaders_train: Optional[TRAIN_DATALOADERS] = None,
                 loaders_val: Optional[EVAL_DATALOADERS] = None,
                 ):
        super().__init__()
        self.loaders_train = loaders_train
        self.loaders_val = loaders_val

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._patch_loaders('train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._patch_loaders('val')

    def _patch_loaders(self, mode: str):
        # if self.trainer.reload_dataloaders_every_n_epochs != 1:
        #     raise AttributeError(f"Invalid parameter for reloading. "
        #                          f"Set {pl.Trainer.__name__}(reload_dataloaders_every_n_epochs=1, ...) to avoid "
        #                          f"using same data on each epoch")

        assert mode in ('train', 'val')
        patcher = self._patch_train_loader_to_ddp if mode == 'train' else self._patch_val_loader_to_ddp
        loaders = self.loaders_train if mode == 'train' else self.loaders_val

        ddp_loaders = patcher(loaders) if isinstance(loaders, DataLoader) else [patcher(loader) for loader in loaders]
        return ddp_loaders

    # def _patch(self, loader: DataLoader, drop_last: bool, shuffle: bool):
    #     if loader.batch_sampler is None and loader.sampler is None:
    #         dataset = loader.dataset
    #         ddp_sampler = DistributedSampler(dataset=dataset, shuffle=shuffle, drop_last=drop_last)
    #         loader = DataLoader(dataset=loader.dataset,
    #                             sampler=ddp_sampler,
    #                             batch_size=loader.batch_size,
    #                             num_workers=loader.num_workers)
    #     elif loader.batch_sampler is not None:
    #         ddp_sampler = DDPSamplerWrapper(sampler=loader.batch_sampler,
    #                                         shuffle=shuffle,
    #                                         drop_last=drop_last)
    #         ddp_sampler.set_epoch(self.trainer.current_epoch)
    #         loader = DataLoader(loader.dataset,
    #                             batch_sampler=ddp_sampler,
    #                             num_workers=loader.num_workers)
    #     elif loader.sampler is not None:
    #         ddp_sampler = DDPSamplerWrapper(sampler=loader.sampler,
    #                                         shuffle=shuffle,
    #                                         drop_last=drop_last)

    def _patch_val_loader_to_ddp(self, loader: DataLoader) -> DataLoader:
        dataset = loader.dataset
        ddp_sampler = DistributedSampler(dataset=dataset, shuffle=False, drop_last=False)
        loader = DataLoader(dataset=loader.dataset,
                            sampler=ddp_sampler,
                            batch_size=loader.batch_size,
                            num_workers=loader.num_workers)
        return loader

    def _patch_train_loader_to_ddp(self, loader: DataLoader) -> DataLoader:
        if loader.batch_sampler is not None:
            ddp_batch_sampler = DDPSamplerWrapper(sampler=loader.batch_sampler,
                                                  shuffle=True,
                                                  drop_last=True)
            ddp_batch_sampler.set_epoch(self.trainer.current_epoch)
            loader = DataLoader(loader.dataset,
                                batch_sampler=ddp_batch_sampler,
                                num_workers=loader.num_workers)
            return loader
        else:
            raise TypeError('NON SUPPORTED')


class RetrievalModule(ModuleDDP):
    def __init__(
        self,
        model: IExtractor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        loaders_train: Optional[TRAIN_DATALOADERS] = None,
        loaders_val: Optional[EVAL_DATALOADERS] = None,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        key_input: str = "input_tensors",
        key_targets: str = "labels",
        key_embeddings: str = "embeddings",
    ):
        super(RetrievalModule, self).__init__(loaders_train=loaders_train, loaders_val=loaders_val)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.key_input = key_input
        self.key_target = key_targets
        self.key_embeddings = key_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(x)
        return embeddings

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        embeddings = self.model(batch[self.key_input])
        bs = len(embeddings)

        loss = self.criterion(embeddings, batch[self.key_target])
        self.log("loss", loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True, sync_dist=True)

        if hasattr(self.criterion, "last_logs"):
            self.log_dict(self.criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False, sync_dist=True)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *dataset_idx: int) -> Dict[str, Any]:
        embeddings = self.model.extract(batch[self.key_input])
        return {**batch, **{self.key_embeddings: embeddings}}

    def configure_optimizers(self) -> Any:
        if self.scheduler is None:
            return self.optimizer
        else:
            scheduler = {
                "scheduler": self.scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
            }
            return [self.optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # https://github.com/Lightning-AI/lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


__all__ = ["RetrievalModule"]
