# type: ignore
# flake8: noqa

from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from oml.const import PATHS_COLUMN
from oml.inference.list_inference import inference_on_images
from oml.interfaces.models import IExtractor, IFreezable, IPairwiseModel
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.utils.misc_torch import elementwise_dist


class PairwiseModule(pl.LightningModule):
    def __init__(
        self,
        pairwise_model,
        pairs_miner,
        optimizer,
        scheduler,
        scheduler_interval,
        scheduler_frequency,
        pair_1st_key,
        pair_2nd_key,
        labels_key,
        embeddings_key,
    ):
        pl.LightningModule.__init__(self)

        self.model = pairwise_model
        self.miner = pairs_miner
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key
        self.labels_key = labels_key
        self.embeddings_key = embeddings_key

        self.criterion = BCEWithLogitsLoss()

    def train_step(self, batch, batch_idx):
        x1 = batch[self.pair_1st_key]
        x2 = batch[self.pair_2nd_key]

        ids1, ids2, is_negative = self.miner.sample(features=batch[self.embeddings_key], labels=batch[self.labels_key])

        predictions = self.model(x1=x2, x2=x2)
        loss = self.criterion(predictions, is_negative.float())

        self.log("loss", loss.item(), prog_bar=True, batch_size=len(x1), on_step=True, on_epoch=True)

        return loss

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


class PairsMiner:
    def __init__(self):
        self.miner = HardTripletsMiner()
        # self.miner = AllTripletsMiner()

    def sample(self, features, labels):
        ii_a, ii_p, ii_n = self.miner._sample(features, labels=labels)

        ii_a_1, ii_p = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_p))))))
        ii_a_2, ii_n = zip(*list(set(list(map(lambda x: tuple(sorted([x[0], x[1]])), zip(ii_a, ii_n))))))

        is_negative = torch.ones(len(ii_a_1) + len(ii_a_2)).bool()
        is_negative[: len(ii_a_1)] = 0

        return torch.tensor([*ii_a_1, *ii_a_2]).long(), torch.tensor([*ii_p, *ii_n]).long(), is_negative


class ImagesSiamese(IPairwiseModel, IFreezable):
    def __init__(self, extractor: IExtractor) -> None:
        super(ImagesSiamese, self).__init__()
        self.extractor = extractor
        feat_dim = self.extractor.feat_dim

        # todo: parametrize
        self.head = nn.Sequential(
            *[
                nn.Linear(feat_dim, feat_dim // 2, bias=True),
                nn.Dropout(),
                nn.Sigmoid(),
                nn.Linear(feat_dim // 2, 1, bias=False),
            ]
        )

        self.train_backbone = True

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.concat([x1, x2], dim=2)

        with torch.set_grad_enabled(self.train_backbone):
            x = self.extractor(x)

        x = self.head(x)
        x = x.squeeze()

        return x

    def freeze(self) -> None:
        self.train_backbone = False

    def unfreeze(self) -> None:
        self.train_backbone = True


class ImagesSiameseTrivial(IPairwiseModel):
    def __init__(self, extractor: IExtractor) -> None:
        super(ImagesSiameseTrivial, self).__init__()
        self.extractor = extractor

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.extractor(x1)
        x2 = self.extractor(x2)
        return elementwise_dist(x1, x2, p=2)


def validate(postprocessor, top_n, df_val, emb_val):
    assert len(df_val) == len(emb_val)

    calculator = EmbeddingMetrics(
        cmc_top_k=(1, postprocessor.top_n),
        precision_top_k=(1, 3, top_n),
        postprocessor=postprocessor,
        extra_keys=("paths",),
    )
    calculator.setup(len(df_val))
    calculator.update_data(
        {
            "embeddings": emb_val,
            "is_query": torch.tensor(df_val["is_query"]).bool(),
            "is_gallery": torch.tensor(df_val["is_gallery"]).bool(),
            "labels": torch.tensor(df_val["label"]).long(),
            "paths": df_val["path"].tolist(),
        }
    )
    metrics = calculator.compute_metrics()
    return metrics


def get_embeddings(
    dataset_root,
    extractor,
    transforms,
    num_workers,
    batch_size,
):
    postfix = getattr(extractor, "weights")
    save_path = Path(dataset_root / f"embeddings_{postfix}.pkl")

    df = pd.read_csv(dataset_root / "df.csv")
    df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: Path(dataset_root) / x)  # todo

    if save_path.is_file():
        embeddings = torch.load(save_path, map_location="cpu")
    else:
        embeddings = inference_on_images(
            model=extractor,
            paths=df[PATHS_COLUMN],
            transform=transforms,
            num_workers=num_workers,
            batch_size=batch_size,
            verbose=True,
        ).cpu()
        torch.save(embeddings, save_path)

    train_mask = df["split"] == "train"

    emb_train = embeddings[train_mask]
    emb_val = embeddings[~train_mask]

    df_train = df[train_mask]
    df_train.reset_index(inplace=True)

    df_val = df[~train_mask]
    df_val.reset_index(inplace=True)

    return emb_train, emb_val, df_train, df_val
