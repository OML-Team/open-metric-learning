import pytorch_lightning as pl
import torch

from examples.cub.convert_cub import build_cub_df
from oml.datasets.retrieval import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.retrieval import RetrievalModule
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import SequentialBalanceSampler

# data
dataset_root = "/nydl/data/CUB_200_2011/"
# download dataset
df = build_cub_df(dataset_root)

# model
model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# train
df_train = df[df["split"] == "train"].reset_index(drop=True)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
train_dataset = DatasetWithLabels(df=df_train, im_size=224, pad_ratio=0.0, transform=None)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = SequentialBalanceSampler(labels=train_dataset.get_labels(), n_labels=8, n_instances=4)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=2 * 4)

# val
df_val = df[df["split"] == "validation"].reset_index(drop=True)
val_dataset = DatasetQueryGallery(df=df_val, im_size=224, pad_ratio=0.0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
metric_callback = MetricValCallback(metric=EmbeddingMetrics())

# run
pl_model = RetrievalModule(model, criterion, optimizer)
trainer = pl.Trainer(max_epochs=1, callbacks=[metric_callback], gpus=[0], num_sanity_val_steps=0)
trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
