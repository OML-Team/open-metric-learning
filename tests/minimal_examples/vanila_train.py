from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from oml.datasets.retrieval import DatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceBatchSampler

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False, use_multi_scale=False, strict_load=True)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

dataset_root = Path("/nydl/data/CUB_200_2011/")
df = pd.read_csv(dataset_root / "df.csv")

train_dataset = DatasetWithLabels(df=df, im_size=224, pad_ratio=0.0, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = BalanceBatchSampler(labels=train_dataset.get_labels(), n_instances=4, n_labels=4)
train_loader = DataLoader(train_dataset, batch_sampler=sampler)

for batch in train_loader:
    embeddings = model(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
