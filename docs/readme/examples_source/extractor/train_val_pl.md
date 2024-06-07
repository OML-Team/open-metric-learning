[comment]:lightning-start
```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam

from oml.datasets import ImageLabeledDataset, ImageQueryGalleryLabeledDataset
from oml.lightning import ExtractorModule
from oml.lightning import MetricValCallback
from oml.losses import ArcFaceLoss
from oml.metrics import EmbeddingMetrics
from oml.models import ViTExtractor
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset
from oml.lightning import logging
from oml.retrieval import ConstantThresholding

df_train, df_val = get_mock_images_dataset(global_paths=True, df_name="df_with_category.csv")

# model
extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=True)

# train
optimizer = Adam(extractor.parameters(), lr=1e-6)
train_dataset = ImageLabeledDataset(df_train)
criterion = ArcFaceLoss(in_features=extractor.feat_dim, num_classes=df_train["label"].nunique())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

# val
val_dataset = ImageQueryGalleryLabeledDataset(df_val)
val_loader = DataLoader(val_dataset, batch_size=4)
metric_callback = MetricValCallback(
    metric=EmbeddingMetrics(dataset=val_dataset, postprocessor=ConstantThresholding(0.8)),
    log_images=True
)

# 1) Logging with Tensorboard
logger = logging.TensorBoardPipelineLogger(".")

# 2) Logging with Neptune
# logger = logging.NeptunePipelineLogger(api_key="", project="", log_model_checkpoints=False)

# 3) Logging with Weights and Biases
# import os
# os.environ["WANDB_API_KEY"] = ""
# logger = logging.WandBPipelineLogger(project="test_project", log_model=False)

# 4) Logging with MLFlow locally
# logger = logging.MLFlowPipelineLogger(experiment_name="exp", tracking_uri="file:./ml-runs")

# 5) Logging with ClearML
# logger = logging.ClearMLPipelineLogger(project_name="exp", task_name="test")

# run
pl_model = ExtractorModule(extractor, criterion, optimizer)
trainer = pl.Trainer(max_epochs=3, callbacks=[metric_callback], num_sanity_val_steps=0, logger=logger)
trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

```
[comment]:lightning-end
