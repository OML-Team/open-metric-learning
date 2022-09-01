from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from oml.const import EMBEDDINGS_KEY, INPUT_TENSORS_KEY, LABELS_KEY
from oml.interfaces.criterions import ICriterion
from oml.interfaces.models import IExtractor


class RetrievalModule(pl.LightningModule):
    def __init__(
        self,
        model: IExtractor,
        criterion: ICriterion,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
    ):
        super(RetrievalModule, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.embeddings_key = embeddings_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(x)
        return embeddings

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        embeddings = self.model(batch[self.input_tensors_key])
        bs = len(embeddings)

        loss = self.criterion(embeddings, batch[self.labels_key])
        loss_name = (getattr(self.criterion, "crit_name", "") + " loss").strip()
        self.log(loss_name, loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        if hasattr(self.criterion, "last_logs"):
            self.log_dict(self.criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False)

        return loss

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
            return [self.optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # https://github.com/Lightning-AI/lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


__all__ = ["RetrievalModule"]
