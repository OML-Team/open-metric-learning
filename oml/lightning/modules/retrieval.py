from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from oml.interfaces.models import IExtractor, IHead


class RetrievalModule(pl.LightningModule):
    def __init__(
        self,
        model: IExtractor,
        head: Optional[IHead],
        criterion: nn.Module,
        clf_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        key_input: str = "input_tensors",
        key_targets: str = "labels",
        key_embeddings: str = "embeddings",
        key_classification: str = "class_predictions",
    ):
        super(RetrievalModule, self).__init__()

        self.model = model
        self.head = head
        self.criterion = criterion
        self.clf_criterion = clf_criterion
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
        self.log("loss", loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        if hasattr(self.criterion, "last_logs"):
            self.log_dict(self.criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False)

        if self.head:
            pred = self.head(embeddings)
            loss_clf = self.clf_criterion(pred, batch[self.key_target])
            self.log("loss_clf", loss_clf.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

            if hasattr(self.clf_criterion, "last_logs"):
                self.log_dict(self.clf_criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

            return loss + loss_clf
        else:
            return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *dataset_idx: int) -> Dict[str, Any]:
        embeddings = self.model(batch[self.key_input])
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
