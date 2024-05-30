<div style="overflow-x: auto;">

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">

<tr>
    <td style="border: 1px solid black; padding: 0;">

Multi GPU

[comment]: https://carbon.now.sh/?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=monokai&wt=none&l=python&width=682&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=2x&wm=false&code=clb%2520%253D%2520MetricValCallback%28EmbeddingMetrics%28...%29%29%250Apl_model%2520%253D%2520ExtractorModuleDDP%28model%252C%2520train_loader%252C%2520val_loader%29%250A%250Atrainer%2520%253D%2520pl.Trainer%28%250A%2520%2520callbacks%253D%255Bclb%255D%252C%2520strategy%253DDDPStrategy%28%29%252C%250A%2520%2520device%253D2%252C%2520use_distributed_sampler%253DFalse%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%29%250Atrainer.fit%28pl_model%29

![](https://i.ibb.co/ryVLnr7/carbon-1.png)

<details>
<summary>Full example</summary>
<p>

[comment]:train-val-img-start
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
[comment]:train-val-img-end

</p>
</details>

</td>

<td style="border: 1px solid black; padding: 0;">

Multi GPU

[comment]: https://carbon.now.sh/?bg=rgba%28171%2C+184%2C+195%2C+1%29&t=monokai&wt=none&l=python&width=682&ds=true&dsyoff=20px&dsblur=68px&wc=true&wa=false&pv=56px&ph=56px&ln=false&fl=1&fm=Hack&fs=14px&lh=133%25&si=false&es=2x&wm=false&code=clb%2520%253D%2520MetricValCallback%28EmbeddingMetrics%28...%29%29%250Apl_model%2520%253D%2520ExtractorModuleDDP%28model%252C%2520train_loader%252C%2520val_loader%29%250A%250Atrainer%2520%253D%2520pl.Trainer%28%250A%2520%2520callbacks%253D%255Bclb%255D%252C%2520strategy%253DDDPStrategy%28%29%252C%250A%2520%2520device%253D2%252C%2520use_distributed_sampler%253DFalse%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%2520%250A%29%250Atrainer.fit%28pl_model%29

![](https://i.ibb.co/ryVLnr7/carbon-1.png)

<details>
<summary>Full example</summary>
<p>

[comment]:train-val-img-start
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
[comment]:train-val-img-end

</p>
</details>


</td>
</tr>
<tr>
<td style="border: 1px solid black; padding: 0;">

TODO

</td>

<td style="border: 1px solid black; padding: 0;">

TODO

</td>

</tr>

</table>

</div>
