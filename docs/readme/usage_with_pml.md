You can easily access a lot of content from [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
with our library. You can see that the examples below are different from the basic ones only in a few lines of code:

<details>
<summary>Training with loss from PML</summary>
<p>

```python
import torch
from tqdm import tqdm

from oml.datasets.base import DatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

from pytorch_metric_learning import losses, distances, reducers, miners

dataset_root = "mock_dataset/"
df_train, _ = download_mock_dataset(dataset_root)

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)

# PML specific
# criterion = losses.TripletMarginLoss(margin=0.2, triplets_per_anchor="all")
criterion = losses.ArcFaceLoss(num_classes=df_train["label"].nunique(), embedding_size=model.feat_dim)  # for classification-like losses

sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = model(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m66v1qhCyAUciEcXsJlIJtjF6nz6ZLI7?usp=sharing)

<details>
<summary>Training with distance, reducer, miner and loss from PML</summary>
<p>

```python
import torch
from tqdm import tqdm

from oml.datasets.base import DatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

from pytorch_metric_learning import losses, distances, reducers, miners

dataset_root = "mock_dataset/"
df_train, _ = download_mock_dataset(dataset_root)

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)

# PML specific
distance = distances.LpDistance(p=2)
reducer = reducers.ThresholdReducer(low=0)
criterion = losses.TripletMarginLoss()
miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all")

sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = model(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"], miner(embeddings, batch["labels"]))  # PML specific
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MbVmSnQvO16eVgAqy1kcOd1XysgaYVBo?usp=sharing)

Note, during the validation process OpenMetricLearning computes *L2* distances. Thus, when choosing a distance from PML,
we recommend you to pick `distances.LpDistance(p=2)`.

To use content from PyTorch Metric Learning with our Config API just follow the standard
[tutorial](https://open-metric-learning.readthedocs.io/en/latest/examples/config.html#how-to-use-my-own-implementation-of-loss-model-augmentations-etc)
of adding custom loss.
