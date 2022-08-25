import numpy as np
import pytorch_lightning as pl
from neptune.new.types import File
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from oml.analysis.visualisation import RetrievalVisualizer
from oml.const import OVERALL_CATEGORIES_KEY
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import flatten_dict


class ImageLoggingCallback(Callback):
    def __init__(
        self,
        metric: EmbeddingMetrics,
        top_k_per_metric: int = 3,
        top_k_in_row: int = 5,
        folder: str = "image_logs",
        log_overall_only: bool = True,
    ) -> None:
        super().__init__()
        self.metric = metric
        assert getattr(metric, "save_non_reduced", False), "You have to save reduced metrics to plot worst cases"
        self.top_k_per_metric = top_k_per_metric
        self.top_k_in_row = top_k_in_row
        self.folder = folder
        self.log_overall_only = log_overall_only

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        visualizer = RetrievalVisualizer.from_embeddings_metric(self.metric)

        for metric_name, metric_values in flatten_dict(self.metric.metrics_unreduced, sep="_").items():  # type: ignore
            if self.log_overall_only and not metric_name.startswith(OVERALL_CATEGORIES_KEY):
                continue
            for n, idx in enumerate(np.argsort(metric_values)[: self.top_k_per_metric]):
                fig = visualizer.visualise(query_idx=idx, top_k=self.top_k_in_row)
                log_str = f"{self.folder}/epoch_{pl_module.current_epoch}/#{n + 1} worst by {metric_name}"
                if isinstance(pl_module.logger, NeptuneLogger):
                    pl_module.logger.experiment[log_str].log(File.as_image(fig))
                elif isinstance(pl_module.logger, TensorBoardLogger):
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    pl_module.logger.experiment.add_image(log_str, np.swapaxes(data, 0, 2), pl_module.current_epoch)
                else:
                    raise ValueError(f"Logging with {type(pl_module.logger)} is not supported yet.")
