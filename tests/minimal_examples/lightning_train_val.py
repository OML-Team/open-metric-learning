from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from oml.datasets.retrieval import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.retrieval import RetrievalModule
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.resnet import ResnetExtractor
from oml.samplers.balance import SequentialBalanceSampler

model = ResnetExtractor(
    weights="resnet50_moco_v2", arch="resnet50", normalise_features=True, gem_p=7.0, remove_fc=True, strict_load=True
)

dataset_root = Path("/nydl/data/CUB_200_2011/")
df = pd.read_csv(dataset_root / "df.csv")

# train
df_train = df[df["split"] == "train"].reset_index(drop=True)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
train_dataset = DatasetWithLabels(df=df_train, im_size=224, pad_ratio=0.0, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = SequentialBalanceSampler(labels=train_dataset.get_labels(), n_labels=2, n_instances=4)
train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=2 * 4)


# val
df_val = df[df["split"] == "validation"].reset_index(drop=True)
val_dataset = DatasetQueryGallery(df=df_val, im_size=224, pad_ratio=0.0, dataset_root=dataset_root)
val_loader = DataLoader(val_dataset, batch_size=8)
metric_callback = MetricValCallback(metric=EmbeddingMetrics())

# run
pl_model = RetrievalModule(model, criterion, optimizer)
trainer = pl.Trainer(max_epochs=1, callbacks=[metric_callback], gpus=[0])
# trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.validate(pl_model, val_loader)
