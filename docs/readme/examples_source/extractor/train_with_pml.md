<details>
<summary>Training with loss from PML</summary>
<p>

```python
import torch
from tqdm import tqdm

from oml.datasets import ImageLabeledDataset
from oml.models import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

from pytorch_metric_learning import losses, distances, reducers, miners

df_train, _ = download_mock_dataset(global_paths=True)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-4)

train_dataset = ImageLabeledDataset(df_train)

# PML specific
# criterion = losses.TripletMarginLoss(margin=0.2, triplets_per_anchor="all")
criterion = losses.ArcFaceLoss(num_classes=df_train["label"].nunique(), embedding_size=extractor.feat_dim)  # for classification-like losses

sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = extractor(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m66v1qhCyAUciEcXsJlIJtjF6nz6ZLI7?usp=sharing)
