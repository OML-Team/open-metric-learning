from itertools import chain
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from oml.const import EMBEDDINGS_KEY, INPUT_TENSORS_KEY, LABELS_KEY
from oml.interfaces.models import IExtractor
from oml.lightning.modules.module_ddp import ModuleDDP
from oml.utils.ddp import sync_dicts_ddp


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
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
        scheduler_monitor_metric: Optional[str] = None,
    ):
        super().__init__(loaders_train=loaders_train, loaders_val=loaders_val)
        # super(RetrievalModule, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.monitor_metric = scheduler_monitor_metric
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.embeddings_key = embeddings_key

    # def training_epoch_end(self, outputs) -> None:
    #     self.check_outputs_of_epoch(outputs, 'train')
    #
    # def validation_epoch_end(self, outputs) -> None:
    #     self.check_outputs_of_epoch(outputs, 'val')

    # def check_outputs_of_epoch(self, outputs) -> None:
    #     # Check point 1 of motivation
    #     world_size = self.trainer.world_size
    #     output_batches = [tuple(out['idx'].tolist()) for out in outputs]
    #     output_batches_synced = sync_dicts_ddp({"batches": output_batches}, world_size)["batches"]
    #
    #     assert len(output_batches_synced) == len(output_batches) * world_size
    #     max_num_not_unique_batches = world_size
    #     assert len(output_batches_synced) - len(set(output_batches_synced)) <= max_num_not_unique_batches

    # def check_outputs_of_epoch(self, outputs, mode) -> None:
    #     # Check point 1 of motivation
    #     world_size = self.trainer.world_size
    #     output_batches = list(chain(*[tuple(out['idx'].tolist()) for out in outputs]))
    #     output_batches_synced = sync_dicts_ddp({"batches": output_batches}, world_size)["batches"]
    #
    #     to_print = ['CHECK', mode, self.global_rank, len(set(output_batches)), len(output_batches), len(set(output_batches_synced)), len(output_batches_synced)]
    #     assert len(output_batches_synced) == len(output_batches) * world_size, to_print
    #     max_num_not_unique_batches = world_size
    #     assert len(output_batches_synced) - len(set(output_batches_synced)) <= max_num_not_unique_batches, to_print

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(x)
        return embeddings

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # print(f'RANK={self.global_rank}', f'{batch_idx=}', f'{self.current_epoch}', sorted(batch['idx'].tolist()))
        embeddings = self.model(batch[self.input_tensors_key])
        bs = len(embeddings)

        loss = self.criterion(embeddings, batch[self.labels_key])
        self.log("loss", loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        if hasattr(self.criterion, "last_logs"):
            self.log_dict(self.criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False)

        return {'loss': loss, 'idx': batch['idx']}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *dataset_idx: int) -> Dict[str, Any]:
        embeddings = self.model.extract(batch[self.input_tensors_key])
        return {**batch, **{self.embeddings_key: embeddings}}

    def configure_optimizers(self) -> Any:
        if self.scheduler is None:
            return self.optimizer
        else:
            scheduler = {
                "scheduler": self.scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
            }
            if isinstance(self.scheduler, ReduceLROnPlateau):
                scheduler["monitor"] = self.monitor_metric
            return [self.optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # https://github.com/Lightning-AI/lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


class RetrievalModuleDDP(RetrievalModule, ModuleDDP):
    def __init__(
            self,
            loaders_train: Optional[TRAIN_DATALOADERS] = None,
            loaders_val: Optional[EVAL_DATALOADERS] = None,
            *args: Any,
            **kwargs: Any
    ):

        RetrievalModule.__init__(self, *args, **kwargs)
        # ModuleDDP.__init__(self, loaders_train=loaders_train, loaders_val=loaders_val)
        # super(RetrievalModule, self).__init__(*args, **kwargs)
        super(ModuleDDP, self).__init__(loaders_train=loaders_train, loaders_val=loaders_val)
        # super(ModuleDDP, self).__init__(loaders_train=loaders_train, loaders_val=loaders_val)


__all__ = ["RetrievalModule", "RetrievalModuleDDP"]
