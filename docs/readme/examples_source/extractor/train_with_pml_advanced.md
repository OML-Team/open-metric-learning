<details>
<summary>Training with distance, reducer, miner and loss from PML</summary>
<p>

```python
from torch.utils.data import DataLoader
from torch.optim import Adam

from oml.datasets import ImageLabeledDataset
from oml.models import ViTExtractor
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset
from oml.transforms.images.torchvision import get_augs_torch

from pytorch_metric_learning import losses, distances, reducers, miners

df_train, _ = get_mock_images_dataset(global_paths=True)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = Adam(extractor.parameters(), lr=1e-4)
train_dataset = ImageLabeledDataset(df_train, transform=get_augs_torch(im_size=224))

# PML specific
distance = distances.LpDistance(p=2)
reducer = reducers.ThresholdReducer(low=0)
criterion = losses.TripletMarginLoss()
miner = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="all")

sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = DataLoader(train_dataset, batch_sampler=sampler)

for batch in train_loader:
    embeddings = extractor(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"], miner(embeddings, batch["labels"]))  # PML specific
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</p>
</details>
