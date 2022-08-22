from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from oml.interfaces.metrics import IBasicMetric
from oml.utils.misc import flatten_dict


class MetricValCallback(Callback):
    def __init__(self, metric: IBasicMetric, loader_idx: int = 0, samples_in_getitem: int = 1):
        """
        It's a wrapper which allows to use IBasicMetric with PyTorch Lightning.

        Args:
            metric: metric
            loader_idx: loader idx
            samples_in_getitem: Some of the datasets return several samples when calling __getitem__,
                so we need to handle it for the proper calculation. For most of the cases this value equals to 1,
                but for TriDataset, which return anchor, positive and negative images, this value must be equal to 3,
                for a dataset of pairs it must be equal to 2.
        """
        self.metric = metric
        self.loader_idx = loader_idx
        self.samples_in_getitem = samples_in_getitem

        self._expected_samples = 0
        self._collected_samples = 0
        self._ready_to_accumulate = False

    def on_validation_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if dataloader_idx == self.loader_idx:
            if not self._ready_to_accumulate:
                len_dataset = len(trainer.val_dataloaders[dataloader_idx].dataset)
                # We use padding in DDP mode, so we need to extend number of expected samples
                len_dataset += len_dataset % trainer.world_size
                len_dataset = len_dataset // trainer.world_size
                self._expected_samples = self.samples_in_getitem * len_dataset
                self._collected_samples = 0

                self.metric.setup(num_samples=self._expected_samples)
                self._ready_to_accumulate = True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if dataloader_idx == self.loader_idx:
            assert self._ready_to_accumulate

            keys_for_metric = self.metric.get_keys_for_metric()
            outputs = {k: v for k, v in outputs.items() if k in keys_for_metric}

            self.metric.update_data(outputs)

            self._collected_samples += len(outputs[list(outputs.keys())[0]])

            if self._collected_samples > self._expected_samples:
                self._raise_computation_error()

            # For non-additive metrics (like f1) we usually accumulate some infortmation during the epoch, then we
            # calculate final score `on_validation_epoch_end`. The problem here is that `on_validation_epoch_end`
            # lightning logger doesn't know dataloader_idx and logs metrics in incorrect way if num_dataloaders > 1.
            # To avoid the problem we calculate metrics `on_validation_batch_end` for the last batch in the loader.
            is_last_expected_batch = self._collected_samples == self._expected_samples
            if is_last_expected_batch:
                # TODO: optimize to avoid duplication of metrics on all devices.
                #  Note: if we calculate metric only on main device, we need to log metric for all devices,
                #  because they need this metric for checkpointing
                self.metric.sync()
                self.calc_and_log_metrics(pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._ready_to_accumulate = False
        if self._collected_samples != self._expected_samples:
            self._raise_computation_error()

    def calc_and_log_metrics(self, pl_module: pl.LightningModule) -> None:
        metrics = self.metric.compute_metrics()
        metrics = flatten_dict(metrics, sep="/")
        pl_module.log_dict(metrics, rank_zero_only=True, add_dataloader_idx=True)

    def _raise_computation_error(self) -> Exception:
        raise ValueError(
            f"Incorrect calculation for {self.metric.__class__.__name__} metric. "
            f"Inconsistent number of samples, obtained: {self._collected_samples}, "
            f"expected: {self._expected_samples}, "
            f"'samples_in_getitem': {self.samples_in_getitem}"
        )


__all__ = ["MetricValCallback"]
