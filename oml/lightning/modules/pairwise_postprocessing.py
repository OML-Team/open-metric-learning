from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

from oml.const import EMBEDDINGS_KEY, INPUT_TENSORS_KEY, LABELS_KEY
from oml.interfaces.models import IFreezable, IPairwiseModel
from oml.lightning.modules.ddp import ModuleDDP
from oml.miners.pairs import PairsMiner


class PairwiseModule(pl.LightningModule):
    def __init__(
        self,
        pairwise_model: IPairwiseModel,
        pairs_miner: PairsMiner,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        input_tensors_key: str = INPUT_TENSORS_KEY,
        labels_key: str = LABELS_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
        scheduler_monitor_metric: Optional[str] = None,
        freeze_n_epochs: int = 0,
    ):
        """

        Args:
            pairwise_model: Pairwise model to train
            pairs_miner: Miner of pairs
            criterion: Criterion to optimize
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scheduler_interval: Interval of calling scheduler (must be ``step`` or ``epoch``)
            scheduler_frequency: Frequency of calling scheduler
            input_tensors_key: Key to get tensors from the batches
            labels_key: Key to get labels from the batches
            embeddings_key: Key to get embeddings from the batches
            scheduler_monitor_metric: Metric to monitor for the schedulers that depend on the metric value
            freeze_n_epochs: number of epochs to freeze model (for n > 0 model has to be a successor of ``IFreezable``
                interface). When ``current_epoch >= freeze_n_epochs`` model is unfreezed. Note that epochs are
                starting with 0.

        """
        pl.LightningModule.__init__(self)

        self.model = pairwise_model
        self.miner = pairs_miner
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.input_tensors_key = input_tensors_key
        self.labels_key = labels_key
        self.embeddings_key = embeddings_key
        self.monitor_metric = scheduler_monitor_metric
        self.freeze_n_epochs = freeze_n_epochs

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        ids1, ids2, is_negative = self.miner.sample(features=batch[self.embeddings_key], labels=batch[self.labels_key])
        x1 = batch[self.input_tensors_key][ids1]
        x2 = batch[self.input_tensors_key][ids2]
        target = is_negative.float()

        predictions = self.model(x1=x1, x2=x2)

        loss = self.criterion(predictions, target.to(predictions.device))

        bs = len(batch[self.labels_key])

        self.log("loss", loss.item(), prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        if self.scheduler is not None:
            self.log("lr", self.scheduler.get_last_lr()[0], prog_bar=True, batch_size=bs, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *_: Any) -> Dict[str, Any]:
        # We simply accumulate batches here since we apply postprocessor during metrics calculation
        return batch

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

    def on_epoch_start(self) -> None:
        if self.freeze_n_epochs and isinstance(self.model, IFreezable):
            if self.current_epoch >= self.freeze_n_epochs:
                self.model.unfreeze()
            else:
                self.model.freeze()

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        # https://github.com/Lightning-AI/lightning/issues/1595
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


class PairwiseModuleDDP(PairwiseModule, ModuleDDP):
    def __init__(
        self,
        loaders_train: Optional[TRAIN_DATALOADERS] = None,
        loaders_val: Optional[EVAL_DATALOADERS] = None,
        *args: Any,
        **kwargs: Any,
    ):
        ModuleDDP.__init__(self, loaders_train=loaders_train, loaders_val=loaders_val)
        PairwiseModule.__init__(self, *args, **kwargs)


__all__ = ["PairwiseModule", "PairwiseModuleDDP"]
