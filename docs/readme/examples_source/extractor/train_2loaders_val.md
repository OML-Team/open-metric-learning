<details>
<summary>Training + 2 loaders validation</summary>
<p>

[comment]:lightning-2loaders-start
```python
import pytorch_lightning as pl
import torch

from oml.datasets.base import DatasetQueryGallery
from oml.losses.triplet import TripletLossWithMiner, AllTripletsMiner
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.extractor import ExtractorModule
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models import ViTExtractor
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
df_train, df_val = download_mock_dataset(dataset_root)

# model
extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

optimizer = torch.optim.SGD(extractor.parameters(), lr=1e-6)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())


# 1st validation dataset
val_dataset_1 = DatasetQueryGallery(df_val, dataset_root=dataset_root,
                                  transform=get_normalisation_resize_torch(im_size=224))
val_loader = torch.utils.data.DataLoader(val_dataset_1, batch_size=4)
metric_callback = MetricValCallback(metric=EmbeddingMetrics(extra_keys=[val_dataset_1.paths_key, ],
                                                            cmc_top_k=[], precision_top_k=[], pfc_variance=[]),
                                    log_images=False, loader_idx=0)

# 2nd validation dataset
val_dataset_2 = DatasetQueryGallery(df_val, dataset_root=dataset_root,
                                    transform=get_normalisation_resize_torch(im_size=224))
val_loader_2 = torch.utils.data.DataLoader(val_dataset_2, batch_size=4)
metric_callback_2 = MetricValCallback(metric=EmbeddingMetrics(extra_keys=[val_dataset_2.paths_key, ],
                                                              cmc_top_k=[], precision_top_k=[], pfc_variance=[]),
                                      log_images=False, loader_idx=1)

# run validation
pl_model = ExtractorModule(extractor, None, None)
trainer = pl.Trainer(max_epochs=3, callbacks=[metric_callback, metric_callback_2], num_sanity_val_steps=0)
metrics_lightning = trainer.validate(pl_model, dataloaders=(val_loader, val_loader_2))

print(metrics_lightning)
print(metric_callback.metric.metrics)
print(metric_callback_2.metric.metrics)
```
[comment]:lightning-2loaders-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bVUgdBGWvQgCkba2YtaIRVlUQUz7Q60Z?usp=share_link)

