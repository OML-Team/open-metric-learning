from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from oml.utils.ddp import patch_dataloader_to_ddp


class ModuleDDP(pl.LightningModule):
    def __init__(
        self,
        loaders_train: Optional[TRAIN_DATALOADERS] = None,
        loaders_val: Optional[EVAL_DATALOADERS] = None,
    ):
        assert loaders_train is not None or loaders_val is not None, "At least one dataloader must be specified"
        pl.LightningModule.__init__(self)
        self.loaders_train = loaders_train
        self.loaders_val = loaders_val

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._patch_loaders("train") if self.loaders_train else super(ModuleDDP, self).train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._patch_loaders("val") if self.loaders_val else super(ModuleDDP, self).val_dataloader()

    def _patch_loaders(self, mode: str) -> EVAL_DATALOADERS:
        assert mode in ("train", "val")
        loaders = self.loaders_train if mode == "train" else self.loaders_val

        if isinstance(loaders, DataLoader):
            return patch_dataloader_to_ddp(loaders)
        elif isinstance(loaders, (list, tuple)):
            return [patch_dataloader_to_ddp(loader) for loader in loaders]
        else:
            raise TypeError("Not supported loaders type")


__all__ = ["ModuleDDP"]
