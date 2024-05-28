
<details>
<summary>Training + Validation [Lightning Distributed]</summary>
<p>

[comment]:lightning-ddp-start
```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam

from oml.datasets import ImageQueryGalleryLabeledDataset, ImageLabeledDataset
from oml.lightning.modules.extractor import ExtractorModuleDDP
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses import TripletLossWithMiner
from oml.metrics import EmbeddingMetrics
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset
from pytorch_lightning.strategies import DDPStrategy
from oml.transforms.images.torchvision import get_augs_torch

df_train, df_val = get_mock_images_dataset(global_paths=True)

# model
extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# train
optimizer = Adam(extractor.parameters(), lr=1e-6)
train_dataset = ImageLabeledDataset(df_train, transform=get_augs_torch(im_size=224))
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

# val
val_dataset = ImageQueryGalleryLabeledDataset(df_val)
val_loader = DataLoader(val_dataset, batch_size=4)
metric_callback = MetricValCallback(metric=EmbeddingMetrics(dataset=val_dataset))

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
