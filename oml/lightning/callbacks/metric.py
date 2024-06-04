import warnings
from math import ceil
from typing import Any, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader

from oml.const import INDEX_KEY, LOG_IMAGE_FOLDER
from oml.ddp.patching import check_loaders_is_patched, patch_dataloader_to_ddp
from oml.interfaces.loggers import IFigureLogger
from oml.interfaces.metrics import IBasicMetric, IMetricVisualisable
from oml.lightning.modules.ddp import ModuleDDP
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
    ):
        """
        Args:
            metric: Metric
            log_images: Set ``True`` if you want to have visual logging
            loader_idx: Idx of the loader to calculate metric for

        """

        self.metric = metric
        self.log_images = log_images
        assert not log_images or (isinstance(metric, IMetricVisualisable) and metric.ready_to_visualize())

        self.loader_idx = loader_idx

        self._expected_samples = 0
        self._collected_samples = 0
        self._ready_to_accumulate = False

    def _calc_expected_samples(self, trainer: pl.Trainer, dataloader_idx: int = 0) -> int:
        loaders = (
            [trainer.val_dataloaders] if isinstance(trainer.val_dataloaders, DataLoader) else trainer.val_dataloaders
        )
        len_dataset = len(loaders[dataloader_idx].dataset)
        if trainer.world_size > 1:
            # we use padding in DDP and sequential sampler for validation
            len_dataset = ceil(len_dataset / trainer.world_size)
        return len_dataset

    def on_validation_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
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
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx == self.loader_idx:
            assert self._ready_to_accumulate

            self.metric.update_data(data=outputs, indices=outputs[INDEX_KEY])  # type: ignore

            self._collected_samples += len(outputs[INDEX_KEY])
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

        if not isinstance(pl_module.logger, IFigureLogger):
            warnings.warn(
                f"Unexpected logger {pl_module.logger}. Figures have not been saved. "
                f"Please, use a child of {IFigureLogger}."
            )
            return

        for fig, metric_log_str in zip(*self.metric.visualize()):
            log_str = f"{LOG_IMAGE_FOLDER}/{metric_log_str}"
            pl_module.logger.log_figure(fig=fig, title=log_str, idx=pl_module.current_epoch)
            plt.close(fig=fig)

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
            f"expected: {self._expected_samples}. "
            f"Make sure that you don't use the 'overfit_batches' parameter in 'pl.Trainer' and "
            f"you set 'drop_last=False'. The idea is that lengths of dataset and dataloader must match."
        )

    @staticmethod
    def _check_loaders(trainer: "pl.Trainer") -> None:
        if trainer.world_size > 1 and trainer.val_dataloaders is not None:
            if not check_loaders_is_patched(trainer.val_dataloaders):
                raise RuntimeError(err_message_loaders_is_not_patched)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._check_loaders(trainer)

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._check_loaders(trainer)


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

__all__ = ["MetricValCallback"]
