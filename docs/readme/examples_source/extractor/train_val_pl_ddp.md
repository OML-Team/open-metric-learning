
<details>
<summary>Training + Validation [Lightning Distributed]</summary>
<p>

[comment]:lightning-ddp-start
```python
import pytorch_lightning as pl
import torch

from oml.datasets import DatasetQueryGallery, DatasetWithLabels
from oml.lightning import ExtractorModuleDDP
from oml.lightning import MetricValCallbackDDP
from oml.losses import TripletLossWithMiner
from oml.metrics import EmbeddingMetricsDDP
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.samplers import BalanceSampler
from oml.utils import download_mock_dataset
from pytorch_lightning.strategies import DDPStrategy

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
metric_callback = MetricValCallbackDDP(metric=EmbeddingMetricsDDP())  # DDP specific

# run
pl_model = ExtractorModuleDDP(extractor=extractor, criterion=criterion, optimizer=optimizer,
                              loaders_train=train_loader, loaders_val=val_loader  # DDP specific
                              )

ddp_args = {"accelerator": "cpu", "devices": 2, "strategy": DDPStrategy(), "use_distributed_sampler": False}  # DDP specific
trainer = pl.Trainer(max_epochs=1, callbacks=[metric_callback], num_sanity_val_steps=0, **ddp_args)
trainer.fit(pl_model)  # we don't pass loaders to .fit() in DDP
```
[comment]:lightning-ddp-end
</p>
</details>

*Colab: there is no Colab link since it provides only single-GPU machines.*
