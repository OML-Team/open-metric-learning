<details>
<summary>Postprocessor: Training + Validation</summary>
<p>

[comment]:postprocessor-start
```python
from pprint import pprint

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from oml.datasets import ImageLabeledDataset, ImageQueryGalleryLabeledDataset, ImageBaseDataset
from oml.inference import inference
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.pairs import PairsMiner
from oml.models import ConcatSiamese, ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.transforms.images.torchvision import get_augs_torch

# In these example we will train a pairwise model as a re-ranker for ViT
extractor = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")
df_train, df_val = download_mock_dataset(global_paths=True)

# SAVE VIT EMBEDDINGS
# - training ones are needed for hard negative sampling when training pairwise model
# - validation ones are needed to construct the original prediction (which we will re-rank)
embeddings_train = inference(extractor, ImageBaseDataset(df_train["path"].tolist(), transform=transforms), batch_size=4, num_workers=0)
embeddings_valid = inference(extractor, ImageBaseDataset(df_val["path"].tolist(), transform=transforms), batch_size=4, num_workers=0)

# TRAIN PAIRWISE MODEL
train_dataset = ImageLabeledDataset(df_train, transform=get_augs_torch(224), extra_data={"embeddings": embeddings_train})
pairwise_model = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])
optimizer = torch.optim.SGD(pairwise_model.parameters(), lr=1e-6)
miner = PairsMiner(hard_mining=True)
criterion = BCEWithLogitsLoss()

train_loader = DataLoader(train_dataset, batch_sampler=BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2))

for batch in train_loader:
    # We sample positive and negative pairs on which the original model struggled most
    ids1, ids2, is_negative_pair = miner.sample(features=batch["embeddings"], labels=batch["labels"])
    probs = pairwise_model(x1=batch["input_tensors"][ids1], x2=batch["input_tensors"][ids2])
    loss = criterion(probs, is_negative_pair.float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# VALIDATE RE-RANKING MODEL
val_dataset = ImageQueryGalleryLabeledDataset(df=df_val, transform=transforms, extra_data={"embeddings": embeddings_valid})
valid_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

postprocessor = PairwiseReranker(top_n=3, pairwise_model=pairwise_model, num_workers=0, batch_size=4)
calculator = EmbeddingMetrics(dataset=val_dataset, postprocessor=postprocessor)
calculator.setup(num_samples=len(val_dataset))

for batch in valid_loader:
    calculator.update_data(data_dict=batch)

pprint(calculator.compute_metrics())  # Pairwise inference happens here
```
[comment]:postprocessor-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LBmusxwo8dPqWznmK627GNMzeDVdjMwv?usp=sharing)
