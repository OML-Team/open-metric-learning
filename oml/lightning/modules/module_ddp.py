from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from oml.const import PolicyDDP
from oml.utils.ddp import patch_dataloader_to_ddp


class ModuleDDP(pl.LightningModule):
    def __init__(self,
                 loaders_train: Optional[TRAIN_DATALOADERS] = None,
                 loaders_val: Optional[EVAL_DATALOADERS] = None,
                 ):
        super().__init__()
        self.loaders_train = loaders_train
        self.loaders_val = loaders_val

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._patch_loaders('train') if self.loaders_train else super(ModuleDDP, self).train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._patch_loaders('val') if self.loaders_val else super(ModuleDDP, self).val_dataloader()

    def _patch_loaders(self, mode: str) -> EVAL_DATALOADERS:
        assert mode in ('train', 'val')

        if mode == 'train':
            loaders = self.loaders_train
            drop_last = PolicyDDP.train_drop_last
            shuffle = PolicyDDP.train_shuffle
        else:
            loaders = self.loaders_val
            drop_last = PolicyDDP.val_drop_last
            shuffle = PolicyDDP.val_shuffle

        if isinstance(loaders, DataLoader):
            return patch_dataloader_to_ddp(loaders, drop_last=drop_last, shuffle=shuffle)
        else:
            return [patch_dataloader_to_ddp(loader, drop_last=drop_last, shuffle=shuffle) for loader in loaders]
