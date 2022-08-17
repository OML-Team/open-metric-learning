from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from oml.interfaces.models import IExtractor


class RetrievalModule(pl.LightningModule):
    def __init__(
        self,
        model: IExtractor,
        emb_criterion: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        clf_criterion: Optional[nn.Module] = None,
        clf_weight: float = 1,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "epoch",
        scheduler_frequency: int = 1,
        input_tensors_key: str = "input_tensors",
        targets_key: str = "labels",
        embeddings_key: str = "embeddings",
    ):
        super(RetrievalModule, self).__init__()

        self.model = model
        self.emb_criterion = emb_criterion
        self.clf_criterion = clf_criterion
        self.clf_weight = clf_weight
        self.optimizer = optimizer

        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency

        self.input_tensors_key = input_tensors_key
        self.targets_key = targets_key
        self.embeddings_key = embeddings_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(x)
        return embeddings

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        embeddings = self.model(batch[self.input_tensors_key])
        bs = len(embeddings)

        if self.emb_criterion is not None:
            loss = self.emb_criterion(embeddings, batch[self.targets_key])
            self.log("loss", loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss = 0

        if hasattr(self.emb_criterion, "last_logs"):
            self.log_dict(self.emb_criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False)

        if self.clf_criterion is not None:
            loss_clf = self.clf_weight * self.clf_criterion(embeddings, batch[self.targets_key])
            self.log("loss_clf", loss_clf.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

            if hasattr(self.clf_criterion, "last_logs"):
                self.log_dict(self.clf_criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

            # return loss + loss_clf  # TODO: fix this
            return loss_clf
        else:
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

    def on_epoch_start(self) -> None:
        if getattr(self.clf_criterion, "renormalize", None):
            self.clf_criterion.renormalize()


__all__ = ["RetrievalModule"]
