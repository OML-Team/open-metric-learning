from math import ceil
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
from neptune.new.types import File
from pytorch_lightning import Callback
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from oml.const import LOG_IMAGE_FOLDER
from oml.ddp.patching import check_loaders_is_patched, patch_dataloader_to_ddp
from oml.interfaces.metrics import IBasicMetric, IMetricDDP, IMetricVisualisable
from oml.lightning.modules.module_ddp import ModuleDDP
from oml.utils.misc import flatten_dict


class MetricValCallback(Callback):
    """
    This is a wrapper which allows to use IBasicMetric with PyTorch Lightning.

    """

    def __init__(
        self,
        metric: IBasicMetric,
        log_images: bool = False,
        loader_idx: int = 0,
        samples_in_getitem: int = 1,
    ):
        """
        Args:
            metric: Metric
            loader_idx: Idx of the loader to calculate metric for
            samples_in_getitem: Some of the datasets return several samples when calling ``__getitem__``,
                so we need to handle it for the proper calculation. For most of the cases this value equals to 1,
                but for the dataset which explicitly return triplets, this value must be equal to 3,
                for a dataset of pairs it must be equal to 2.

        """

        self.metric = metric
        self.log_images = log_images
        assert not log_images or (
            isinstance(metric, IMetricVisualisable) and metric.ready_to_visualize()  # type: ignore
        )

        self.loader_idx = loader_idx
        self.samples_in_getitem = samples_in_getitem

        self._expected_samples = 0
        self._collected_samples = 0
        self._ready_to_accumulate = False

    def _calc_expected_samples(self, trainer: pl.Trainer, dataloader_idx: int) -> int:
        return self.samples_in_getitem * len(trainer.val_dataloaders[dataloader_idx].dataset)

    def on_validation_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if dataloader_idx == self.loader_idx:
            if not self._ready_to_accumulate:
                self._expected_samples = self._calc_expected_samples(trainer=trainer, dataloader_idx=dataloader_idx)
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

            self.metric.update_data(outputs)

            self._collected_samples += len(outputs[list(outputs.keys())[0]])
            if self._collected_samples > self._expected_samples:
                self._raise_computation_error()

            # For non-additive metrics (like f1) we usually accumulate some information during the epoch, then we
            # calculate final score `on_validation_epoch_end`. The problem here is that `on_validation_epoch_end`
            # lightning logger doesn't know dataloader_idx and logs metrics in incorrect way if num_dataloaders > 1.
            # To avoid the problem we calculate metrics `on_validation_batch_end` for the last batch in the loader.
            is_last_expected_batch = self._collected_samples == self._expected_samples
            if is_last_expected_batch:
                self.calc_and_log_metrics(pl_module)

    def _log_images(self, pl_module: pl.LightningModule) -> None:
        if not isinstance(self.metric, IMetricVisualisable):
            return

        for fig, metric_log_str in zip(*self.metric.visualize()):
            log_str = f"{LOG_IMAGE_FOLDER}/{metric_log_str}"
            if isinstance(pl_module.logger, NeptuneLogger):
                pl_module.logger.experiment[log_str].log(File.as_image(fig))
            elif isinstance(pl_module.logger, TensorBoardLogger):
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                pl_module.logger.experiment.add_image(log_str, np.swapaxes(data, 0, 2), pl_module.current_epoch)
            else:
                raise ValueError(f"Logging with {type(pl_module.logger)} is not supported yet.")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._ready_to_accumulate = False
        if self._collected_samples != self._expected_samples:
            self._raise_computation_error()

    def calc_and_log_metrics(self, pl_module: pl.LightningModule) -> None:
        metrics = self.metric.compute_metrics()
        metrics = flatten_dict(metrics)
        pl_module.log_dict(metrics, rank_zero_only=True, add_dataloader_idx=True)

        if self.log_images:
            self._log_images(pl_module=pl_module)

    def _raise_computation_error(self) -> Exception:
        raise ValueError(
            f"Incorrect calculation for {self.metric.__class__.__name__} metric. "
            f"Inconsistent number of samples, obtained: {self._collected_samples}, "
            f"expected: {self._expected_samples}, "
            f"'samples_in_getitem': {self.samples_in_getitem}.\n"
            f"Make sure that you don't use the 'overfit_batches' parameter in 'pl.Trainer' and "
            f"you set 'drop_last=False'. The idea is that lengths of dataset and dataloader must match."
        )


err_message_loaders_is_not_patched = (
    "\nExperiment is runned in DDP mode, but some of validation dataloaders is not patched. Metric callback will "
    "be incorrect  without patched loaders. Possible problems and solutions:\n"
    f"1) If you use custom module inherited from 'pl.LightningModule', please replace ancestor class with "
    f"our '{ModuleDDP.__name__}', which automaticaly patches your loaders\n"
    f"2) If you implement your own 'train_dataloader' or 'val_dataloader' methods for your module, you can add "
    f"extra line of code for patching loader with '{patch_dataloader_to_ddp.__name__}' function\n"
    f"3) If you call 'trainer.fit(...)' or 'trainer.validate(...)' method with loaders as argument, PytorchLightning "
    f"will ignore loaders from 'train_dataloader' and 'val_dataloader' methods. Please avoid substituting loaders to "
    f"this functions, instead use '{ModuleDDP.__name__}'\n"
    f"4) Check that the flag 'replace_sampler_ddp=False' in the trainer constructor, because we do this "
    f"replacement in '{ModuleDDP.__name__}' constructor\n"
    f"5) Turn off the 'overfit_batches' parameter in 'pl.Trainer'."
)


class MetricValCallbackDDP(MetricValCallback):
    """
    This is an extension to the regular callback that takes into account data reduction and padding
    on the inference for each device in DDP setup

    """

    metric: IMetricDDP

    def __init__(self, metric: IMetricDDP, *args: Any, **kwargs: Any):
        assert isinstance(metric, IMetricDDP), "Metric has to support DDP interface"
        super().__init__(metric, *args, **kwargs)

    def _calc_expected_samples(self, trainer: pl.Trainer, dataloader_idx: int) -> int:
        len_dataset = len(trainer.val_dataloaders[dataloader_idx].dataset)
        if trainer.world_size > 1:
            # we use padding in DDP and sequential sampler for validation
            len_dataset = ceil(len_dataset / trainer.world_size)
        return self.samples_in_getitem * len_dataset

    def calc_and_log_metrics(self, pl_module: pl.LightningModule) -> None:
        # TODO: optimize to avoid duplication of metrics on all devices.
        #  Note: if we calculate metric only on main device, we need to log (!!!) metric for all devices,
        #  because they need this metric for checkpointing
        self.metric.sync()
        return super().calc_and_log_metrics(pl_module=pl_module)

    @staticmethod
    def _check_loaders(trainer: "pl.Trainer") -> None:
        if trainer.world_size > 1:
            if not check_loaders_is_patched(trainer.val_dataloaders):
                raise RuntimeError(err_message_loaders_is_not_patched)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._check_loaders(trainer)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._check_loaders(trainer)


__all__ = ["MetricValCallback", "MetricValCallbackDDP"]
