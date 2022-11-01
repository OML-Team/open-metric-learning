from typing import Dict, Optional, Sequence, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from oml.ddp.patching import patch_dataloader_to_ddp

TValDataloaders = Union[DataLoader, Sequence[DataLoader]]
TTrainDataloaders = Union[TValDataloaders, Dict[str, DataLoader]]


class ModuleDDP(pl.LightningModule):
    """
    The module automatically patches training and validation dataloaders to DDP mode by splitting available indices
    between devices. Note, don't use ``trainer.fit(...)`` or ``trainer.validate(...)``, because in this case,
    `PytorchLightning` will ignore our patching.

    """

    def __init__(
        self,
        loaders_train: Optional[TTrainDataloaders] = None,
        loaders_val: Optional[TValDataloaders] = None,
    ):
        assert loaders_train is not None or loaders_val is not None, "At least one dataloader must be specified"
        pl.LightningModule.__init__(self)
        self.loaders_train = loaders_train
        self.loaders_val = loaders_val

    def train_dataloader(self) -> TTrainDataloaders:
        return self._patch_loaders("train") if self.loaders_train else super(ModuleDDP, self).train_dataloader()

    def val_dataloader(self) -> TValDataloaders:
        return self._patch_loaders("val") if self.loaders_val else super(ModuleDDP, self).val_dataloader()

    def _patch_loaders(self, mode: str) -> TTrainDataloaders:
        assert mode in ("train", "val")
        loaders = self.loaders_train if mode == "train" else self.loaders_val

        if isinstance(loaders, DataLoader):
            return patch_dataloader_to_ddp(loaders)
        elif isinstance(loaders, (list, tuple)):
            return [patch_dataloader_to_ddp(loader) for loader in loaders]
        elif isinstance(loaders, dict):
            return {k: patch_dataloader_to_ddp(loader) for k, loader in loaders.items()}
        else:
            raise TypeError("Not supported loaders type")


__all__ = ["ModuleDDP"]
