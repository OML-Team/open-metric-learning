from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from torch import nn


class RetrievalModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        key_input: str = "input_tensors",
        key_targets: str = "labels",
        key_embeddings: str = "embeddings",
    ):
        super(RetrievalModule, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.key_input = key_input
        self.key_target = key_targets
        self.key_embeddings = key_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(x)
        return embeddings

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        return [self.optimizer]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        embeddings = self.model(batch[self.key_input])

        loss = self.criterion(embeddings, batch[self.key_target])

        args = {"prog_bar": True, "batch_size": len(embeddings), "on_step": True, "on_epoch": True}
        self.log("loss", loss.item(), **args)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *dataset_idx: int) -> Dict[str, Any]:
        embeddings = self.model(batch[self.key_input])
        return {**batch, **{self.key_embeddings: embeddings}}
