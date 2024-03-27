<details>
<summary>Training + Validation [Lightning and logging]</summary>
<p>

[comment]:lightning-start
```python
import pytorch_lightning as pl
import torch

from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.modules.extractor import ExtractorModule
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.lightning.pipelines.logging import NeptunePipelineLogger, TensorBoardPipelineLogger, WandBPipelineLogger

dataset_root = "mock_dataset/"
df_train, df_val = download_mock_dataset(dataset_root)

# model
extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# train
optimizer = torch.optim.SGD(extractor.parameters(), lr=1e-6)
train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)

# val
val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
metric_callback = MetricValCallback(metric=EmbeddingMetrics(extra_keys=[train_dataset.paths_key,]), log_images=True)

# 1) Logging with Tensorboard
logger = TensorBoardPipelineLogger(".")

# 2) Logging with Neptune
# logger = NeptunePipelineLogger(api_key="", project="", log_model_checkpoints=False)

# 3) Logging with Weights and Biases
# import os
# os.environ["WANDB_API_KEY"] = ""
# logger = WandBPipelineLogger(project="test_project", log_model=False)

# run
pl_model = ExtractorModule(extractor, criterion, optimizer)
trainer = pl.Trainer(max_epochs=3, callbacks=[metric_callback], num_sanity_val_steps=0, logger=logger)
trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

```
[comment]:lightning-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bVUgdBGWvQgCkba2YtaIRVlUQUz7Q60Z?usp=share_link)
