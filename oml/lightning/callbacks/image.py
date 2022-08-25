import numpy as np
import pytorch_lightning as pl
from neptune.new.types import File
from pytorch_lightning.callbacks import Callback

from oml.analysis.visualisation import RetrievalVisualizer
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import flatten_dict


class ImageLoggingCallback(Callback):
    def __init__(self, metric: EmbeddingMetrics, top_k_per_metric: int = 3, top_k_in_row: int = 5) -> None:
        super().__init__()
        self.metric = metric
        assert getattr(metric, "save_non_reduced", False), "You have to save reduced metrics to plot worst cases"
        self.top_k_per_metric = top_k_per_metric
        self.top_k_in_row = top_k_in_row

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        visualizer = RetrievalVisualizer.from_embeddings_metric(self.metric)

        for metric_name, metric_values in flatten_dict(self.metric.metrics_unreduced).items():  # type: ignore
            for n, idx in enumerate(np.argsort(metric_values)[: self.top_k_per_metric]):
                fig = visualizer.visualise(query_idx=idx, top_k=self.top_k_in_row)
                pl_module.logger.experiments[f"#{n + 1} worst by {metric_name}"].log(File.as_image(fig))
