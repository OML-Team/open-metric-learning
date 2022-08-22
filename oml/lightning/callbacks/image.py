from typing import Any, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MyPrintingCallback(Callback):
    def __init__(self, log_interval: int = 10, input_tensors_key: str = "input_tensors") -> None:
        super().__init__()
        self.log_interval = log_interval
        self.input_tensors_key = input_tensors_key

    def _plot_batch(self, x: Any) -> plt.Figure:  # noqa
        pass

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if batch_idx and batch_idx % self.log_interval == 0:
            batch[self.input_tensors_key].detach().cpu().numpy()
            pl_module.log("lr")
