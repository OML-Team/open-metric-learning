<details>
<summary>Training with loss from PML</summary>
<p>

```python
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml.datasets import ImageLabeledDataset
from oml.models import ViTExtractor
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset

from pytorch_metric_learning import losses

df_train, _ = get_mock_images_dataset(global_paths=True)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = Adam(extractor.parameters(), lr=1e-4)

train_dataset = ImageLabeledDataset(df_train)

# PML specific
# criterion = losses.TripletMarginLoss(margin=0.2, triplets_per_anchor="all")
criterion = losses.ArcFaceLoss(num_classes=df_train["label"].nunique(), embedding_size=extractor.feat_dim)  # for classification-like losses

sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = DataLoader(train_dataset, batch_sampler=sampler)

for batch in train_loader:
    embeddings = extractor(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</p>
</details>
