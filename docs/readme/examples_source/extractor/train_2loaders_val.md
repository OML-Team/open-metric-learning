[comment]:lightning-2loaders-start
```python
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from oml.datasets import ImageQueryGalleryLabeledDataset
from oml.lightning.callbacks.metric import MetricValCallback
from oml.lightning.modules.extractor import ExtractorModule
from oml.metrics import EmbeddingMetrics
from oml.models import ViTExtractor
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.utils import get_mock_images_dataset

_, df_val = get_mock_images_dataset(global_paths=True)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# 1st validation dataset (big images)
val_dataset_1 = ImageQueryGalleryLabeledDataset(df_val, transform=get_normalisation_resize_torch(im_size=224))
val_loader_1 = DataLoader(val_dataset_1, batch_size=4)
metric_callback_1 = MetricValCallback(
    metric=EmbeddingMetrics(dataset=val_dataset_1), log_images=True, loader_idx=0
)

# 2nd validation dataset (small images)
val_dataset_2 = ImageQueryGalleryLabeledDataset(df_val, transform=get_normalisation_resize_torch(im_size=48))
val_loader_2 = DataLoader(val_dataset_2, batch_size=4)
metric_callback_2 = MetricValCallback(
    metric=EmbeddingMetrics(dataset=val_dataset_2), log_images=True, loader_idx=1
)

# run validation
pl_model = ExtractorModule(extractor, None, None)
trainer = pl.Trainer(max_epochs=3, callbacks=[metric_callback_1, metric_callback_2], num_sanity_val_steps=0)
trainer.validate(pl_model, dataloaders=(val_loader_1, val_loader_2))

print(metric_callback_1.metric.retrieval_results)
print(metric_callback_2.metric.retrieval_results)
```
[comment]:lightning-2loaders-end
