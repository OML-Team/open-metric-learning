from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import nn
from torch.distributed import get_rank
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from oml.const import EMBEDDINGS_KEY, INPUT_TENSORS_KEY, LABELS_KEY
from oml.ddp.utils import is_main_process
from oml.interfaces.criterions import ICriterion
from oml.interfaces.models import IExtractor
from oml.lightning.modules.module_ddp import ModuleDDP


class RetrievalModule(pl.LightningModule):
    """
    This is a base module to train your model with Lightning.

    """

    def __init__(
        self,
        model: IExtractor,
        clf_criterion: Optional[ICriterion],
        emb_criterion: Optional[ICriterion],
        optimizer: torch.optim.Optimizer,
        clf_loss_weight: float = 0.1,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
        scheduler_monitor_metric: Optional[str] = None,
    ):
        """

        Args:
            model: Model to train
            criterion: Criterion to optimize
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scheduler_interval: Interval of calling scheduler (must be ``step`` or ``epoch``)
            scheduler_frequency: Frequency of calling scheduler
            input_tensors_key: Key to get tensors from the batches
            labels_key: Key to get labels from the batches
            embeddings_key: Key to get embeddings from the batches
            scheduler_monitor_metric: Metric to monitor for the schedulers that depend on the metric value

        """
        pl.LightningModule.__init__(self)

        assert clf_criterion or emb_criterion, "You need at least one criterion!"

        self.model = model
        self.clf_criterion = clf_criterion
        self.emb_criterion = emb_criterion
        if emb_criterion and clf_criterion:
            self.clf_loss_weight = clf_loss_weight  # TODO: DOCSTRING!!!
        elif emb_criterion:
            self.clf_loss_weight = 0
        else:  # if clf_criterion
            self.clf_loss_weight = 1
        self.optimizer = optimizer

        self.monitor_metric = scheduler_monitor_metric
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

        loss_emb = loss_clf = 0
        if self.current_epoch > 10 and self.clf_criterion and hasattr(self.clf_criterion, "detach"):
            self.clf_criterion.detach = False
        if self.clf_criterion:
            loss_clf = self.clf_criterion(embeddings, batch[self.labels_key])
            loss_name = (getattr(self.clf_criterion, "crit_name", "") + " loss").strip()
            self.log(loss_name, loss_clf.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)
        if self.emb_criterion:
            loss_emb = self.emb_criterion(embeddings, batch[self.labels_key])
            loss_name = (getattr(self.emb_criterion, "crit_name", "") + " loss").strip()
            self.log(loss_name, loss_emb.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        for criterion in [self.clf_criterion, self.emb_criterion]:
            if hasattr(criterion, "last_logs"):
                if "accuracy" in criterion.last_logs:
                    self.log(
                        "accuracy", criterion.last_logs.pop("accuracy"), prog_bar=False, on_step=False, on_epoch=True
                    )
                self.log_dict(criterion.last_logs, prog_bar=False, batch_size=bs, on_step=True, on_epoch=False)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=False)

        loss = self.clf_loss_weight * loss_clf + (1 - self.clf_loss_weight) * loss_emb
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
            if isinstance(self.scheduler, ReduceLROnPlateau):
                scheduler["monitor"] = self.monitor_metric
            return [self.optimizer], [scheduler]

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # https://github.com/Lightning-AI/lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


class RetrievalModuleDDP(RetrievalModule, ModuleDDP):
    """
    This is a base module for the training of your model with Lightning in DDP.

    """

    def __init__(
        self,
        loaders_train: Optional[TRAIN_DATALOADERS] = None,
        loaders_val: Optional[EVAL_DATALOADERS] = None,
        *args: Any,
        **kwargs: Any
    ):
        ModuleDDP.__init__(self, loaders_train=loaders_train, loaders_val=loaders_val)
        RetrievalModule.__init__(self, *args, **kwargs)


__all__ = ["RetrievalModule", "RetrievalModuleDDP"]
