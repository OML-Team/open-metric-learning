from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

from oml.const import PolicyDDP
from oml.interfaces.metrics import IBasicMetric
from oml.lightning.modules.module_ddp import ModuleDDP
from oml.utils.ddp import check_loaders_is_patched, patch_dataloader_to_ddp
from oml.utils.misc import flatten_dict


err_message_loaders_is_not_patched = \
    "\nExperiment is runned in DDP mode, but some of dataloaders is not patched. Metric callback will be incorrect " \
    "without patched loaders. Possible problems and solutions:\n" \
    f"1) If you use custom module inherited from '{LightningModule.__name__}', please replace ancestor class with " \
    f"our '{ModuleDDP.__name__}', which automaticaly patches your loaders\n" \
    f"2) If you implement your own '{LightningModule.train_dataloader.__name__}' or " \
    f"'{LightningModule.val_dataloader.__name__}' methods for your module, you can add extra line of code for " \
    f"patching loader with '{patch_dataloader_to_ddp.__name__}' function\n" \
    f"3) If you call 'trainer.{Trainer.fit.__name__}(...)' or 'trainer.{Trainer.validate.__name__}(...)' method with " \
    f"loaders as argument, PytorchLightning will ignore loaders from '{LightningModule.train_dataloader.__name__}' " \
    f"and '{LightningModule.val_dataloader.__name__}' methods. Please avoid substituting loaders to this functions, " \
    f"instead use '{ModuleDDP.__name__}'\n" \
    f"4) Check that the flag 'replace_sampler_ddp=False' in the trainer constructor, because we do this " \
    f"replacement in '{ModuleDDP.__name__}' constructor"


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

        self._loaders_checked = False

    def _check_loaders(self, trainer: "pl.Trainer"):
        if not self._loaders_checked:
            if trainer.world_size != 1:
                if not check_loaders_is_patched(trainer.val_dataloaders):
                    raise RuntimeError(err_message_loaders_is_not_patched)

            self._loaders_checked = True

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._check_loaders(trainer)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._check_loaders(trainer)

    def on_validation_batch_start(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if dataloader_idx == self.loader_idx:
            if not self._ready_to_accumulate:
                len_dataset = len(trainer.val_dataloaders[dataloader_idx].dataset)
                if not PolicyDDP.val_drop_last:
                    # If we use padding in DDP mode, we need to extend number of expected samples
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
                #  Note: if we calculate metric only on main device, we need to log (!!!) metric for all devices,
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
