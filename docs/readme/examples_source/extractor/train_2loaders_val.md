<details>
<summary>Validation with 2 loaders</summary>
<p>

[comment]:lightning-2loaders-start
```python
import pytorch_lightning as pl
import torch

from oml.datasets.base import DatasetQueryGallery
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.extractor import ExtractorModule
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models import ViTExtractor
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
_, df_val = download_mock_dataset(dataset_root)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# 1st validation dataset (big images)
val_dataset_1 = DatasetQueryGallery(df_val, dataset_root=dataset_root,
                                    transform=get_normalisation_resize_torch(im_size=224))
val_loader_1 = torch.utils.data.DataLoader(val_dataset_1, batch_size=4)
metric_callback_1 = MetricValCallback(metric=EmbeddingMetrics(extra_keys=[val_dataset_1.paths_key,]),
                                      log_images=True, loader_idx=0)

# 2nd validation dataset (small images)
val_dataset_2 = DatasetQueryGallery(df_val, dataset_root=dataset_root,
                                    transform=get_normalisation_resize_torch(im_size=48))
val_loader_2 = torch.utils.data.DataLoader(val_dataset_2, batch_size=4)
metric_callback_2 = MetricValCallback(metric=EmbeddingMetrics(extra_keys=[val_dataset_2.paths_key,]),
                                      log_images=True, loader_idx=1)

# run validation
pl_model = ExtractorModule(extractor, None, None)
trainer = pl.Trainer(max_epochs=3, callbacks=[metric_callback_1, metric_callback_2], num_sanity_val_steps=0)
trainer.validate(pl_model, dataloaders=(val_loader_1, val_loader_2))

print(metric_callback_1.metric.metrics)
print(metric_callback_2.metric.metrics)
```
[comment]:lightning-2loaders-end
</p>
</details>

<p></p>
