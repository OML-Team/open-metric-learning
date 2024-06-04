```bash
pip install transformers
```

[comment]:lightning-ddp-start
```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam

from oml import datasets as d
from oml.lightning import ExtractorModuleDDP
from oml.lightning import MetricValCallback
from oml.losses import TripletLossWithMiner
from oml.metrics import EmbeddingMetrics
from oml.miners import AllTripletsMiner
from oml.models import HFWrapper
from oml.samplers import BalanceSampler
from oml.utils import get_mock_texts_dataset
from pytorch_lightning.strategies import DDPStrategy

from transformers import AutoModel, AutoTokenizer

df_train, df_val = get_mock_texts_dataset()

# model
extractor = HFWrapper(AutoModel.from_pretrained("bert-base-uncased"), 768)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# train
optimizer = Adam(extractor.parameters(), lr=1e-6)
train_dataset = d.TextLabeledDataset(df_train, tokenizer=tokenizer)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

# val
val_dataset = d.TextQueryGalleryLabeledDataset(df_val, tokenizer=tokenizer)
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
