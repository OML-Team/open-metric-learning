from typing import Any, Iterable, Optional

import numpy as np
import pytorch_lightning as pl
from neptune.new.types import File
from pytorch_lightning import Callback
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from oml.const import (
    LOG_IMAGE_FOLDER,
    LOG_TOPK_IMAGES_PER_ROW,
    LOG_TOPK_ROWS_PER_METRIC,
    OVERALL_CATEGORIES_KEY,
)
from oml.interfaces.metrics import IBasicMetric
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import flatten_dict


class MetricValCallback(Callback):
    def __init__(
        self,
        metric: IBasicMetric,
        save_image_logs: bool = False,
        log_only_main_category: bool = True,
        loader_idx: int = 0,
        samples_in_getitem: int = 1,
        image_metrics_to_ignore: Iterable[str] = ("cmc",),
    ):
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
        self.save_image_logs = save_image_logs

        assert (
            getattr(metric, "save_non_reduced", False) or not self.save_image_logs
        ), "You have to save reduced metrics to plot worst cases"

        self.log_only_main_category = log_only_main_category
        self.loader_idx = loader_idx
        self.samples_in_getitem = samples_in_getitem

        self.image_metrics_to_ignore = image_metrics_to_ignore

        self._expected_samples = 0
        self._collected_samples = 0
        self._ready_to_accumulate = False

    def on_validation_batch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if dataloader_idx == self.loader_idx:
            if not self._ready_to_accumulate:
                self._expected_samples = self.samples_in_getitem * len(trainer.val_dataloaders[dataloader_idx].dataset)
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

    def _log_images(self, pl_module: pl.LightningDataModule) -> None:
        if not isinstance(self.metric, EmbeddingMetrics):
            return

        for metric_name in flatten_dict(self.metric.metrics):  # type: ignore
            if any([ignore in metric_name for ignore in self.image_metrics_to_ignore]):
                continue
            if self.log_only_main_category and not metric_name.startswith(OVERALL_CATEGORIES_KEY):
                continue
            fig = self.metric.get_plot_for_worst_queries(
                metric_name=metric_name, topk_queries=LOG_TOPK_ROWS_PER_METRIC, topk_instances=LOG_TOPK_IMAGES_PER_ROW
            )
            metric_name_for_logging = metric_name.replace("/", "_")
            log_str = f"{LOG_IMAGE_FOLDER}/epoch_{pl_module.current_epoch}/top {LOG_TOPK_ROWS_PER_METRIC} worst by {metric_name_for_logging}"
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

        if self.log_only_main_category:
            metrics = {self.metric.overall_categories_key: metrics[self.metric.overall_categories_key]}

        metrics = flatten_dict(metrics)
        pl_module.log_dict(metrics, rank_zero_only=True, add_dataloader_idx=True)

        if self.save_image_logs:
            self._log_images(pl_module=pl_module)

    def _raise_computation_error(self) -> Exception:
        raise ValueError(
            f"Incorrect calculation for {self.metric.__class__.__name__} metric. "
            f"Inconsistent number of samples, obtained: {self._collected_samples}, "
            f"expected: {self._expected_samples}, "
            f"'samples_in_getitem': {self.samples_in_getitem}"
        )


__all__ = ["MetricValCallback"]
