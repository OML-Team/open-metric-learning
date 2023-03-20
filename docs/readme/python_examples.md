The most flexible, but knowledge-requiring approach.
You are not limited by our project structure and you can use only that part of the functionality which you need.
You can start with fully working code snippets below that train and validate the model
on a tiny dataset of
[figures](https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing).
ㅤ

**Feature extractor**

<details>
<summary>Training</summary>
<p>

[comment]:vanilla-train-start
```python
import torch
from tqdm import tqdm

from oml.datasets.base import DatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
df_train, _ = download_mock_dataset(dataset_root)

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = model(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
[comment]:vanilla-train-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kntDAIdIZ9L40jcndguLAb-XqmCFOgS5?usp=sharing)

<details>
<summary>Validation</summary>
<p>

[comment]:vanilla-validation-start
```python
import torch
from tqdm import tqdm

from oml.datasets.base import DatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models.vit.vit import ViTExtractor
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root =  "mock_dataset/"
_, df_val = download_mock_dataset(dataset_root)

model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).eval()

val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
calculator = EmbeddingMetrics()
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        batch["embeddings"] = model(batch["input_tensors"])
        calculator.update_data(batch)

metrics = calculator.compute_metrics()
```
[comment]:vanilla-validation-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O2o3k8I8jN5hRin3dKnAS3WsgG04tmIT?usp=sharing)

<details>
<summary>Training + Validation [Lightning]</summary>
<p>

[comment]:lightning-start
```python
import pytorch_lightning as pl
import torch

from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.modules.retrieval import RetrievalModule
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root =  "mock_dataset/"
df_train, df_val = download_mock_dataset(dataset_root)

# model
model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# train
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)

# val
val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
metric_callback = MetricValCallback(metric=EmbeddingMetrics())

# run
pl_model = RetrievalModule(model, criterion, optimizer)
trainer = pl.Trainer(max_epochs=1, callbacks=[metric_callback], num_sanity_val_steps=0)
trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```
[comment]:lightning-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bVUgdBGWvQgCkba2YtaIRVlUQUz7Q60Z?usp=share_link)

<details>
<summary>Training + Validation [Lightning Distributed]</summary>
<p>

[comment]:lightning-ddp-start
```python
import pytorch_lightning as pl
import torch

from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.modules.retrieval import RetrievalModuleDDP
from oml.lightning.callbacks.metric import MetricValCallbackDDP
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetricsDDP
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
df_train, df_val = download_mock_dataset(dataset_root)

# model
model = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# train
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)

# val
val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
metric_callback = MetricValCallbackDDP(metric=EmbeddingMetricsDDP())  # DDP specific

# run
pl_model = RetrievalModuleDDP(model=model, criterion=criterion, optimizer=optimizer,
                              loaders_train=train_loader, loaders_val=val_loader  # DDP specific
                              )

ddp_args = {"accelerator": "auto", "devices": 2, "strategy": pl.plugins.DDPPlugin(), "replace_sampler_ddp": False} # DDP specific
trainer = pl.Trainer(max_epochs=1, callbacks=[metric_callback], num_sanity_val_steps=0, **ddp_args)
trainer.fit(pl_model)  # we don't pass loaders to .fit() in DDP
```
[comment]:lightning-ddp-end
</p>
</details>

*Colab: there is no Colab link since it provides only single-GPU machines.*
ㅤ

<details>
<summary>Using a trained model for retrieval</summary>
<p>

[comment]:usage-retrieval-start
```python
import torch

from oml.const import MOCK_DATASET_PATH
from oml.inference.flat import inference_on_images
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc_torch import pairwise_dist

_, df_val = download_mock_dataset(MOCK_DATASET_PATH)
df_val["path"] = df_val["path"].apply(lambda x: MOCK_DATASET_PATH / x)
queries = df_val[df_val["is_query"]]["path"].tolist()
galleries = df_val[df_val["is_gallery"]]["path"].tolist()

model = ViTExtractor.from_pretrained("vits16_dino")
transform, _ = get_transforms_for_pretrained("vits16_dino")

args = {"num_workers": 0, "batch_size": 8}
features_queries = inference_on_images(model, paths=queries, transform=transform, **args)
features_galleries = inference_on_images(model, paths=galleries, transform=transform, **args)

# Now we can explicitly build pairwise matrix of distances or save you RAM via using kNN
use_knn = True
top_k = 3

if use_knn:
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(algorithm="auto", p=2)
    knn.fit(features_galleries)
    dists, ii_closest = knn.kneighbors(features_queries, n_neighbors=top_k, return_distance=True)

else:
    dist_mat = pairwise_dist(x1=features_queries, x2=features_galleries)
    dists, ii_closest = torch.topk(dist_mat, dim=1, k=top_k, largest=False)

print(f"Top {top_k} items closest to queries are:\n {ii_closest}")
```
[comment]:usage-retrieval-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S2nK6KaReDm-RjjdojdId6CakhhSyvfA?usp=share_link)
ㅤ

**Postprocessing**

You can also boost retrieval accuracy of your features extractor by adding a postprocessor (we recommend
to check the examples above first).
In the example below we train a siamese model to re-rank top retrieval outputs of the original model
by performing inference on pairs ``(query, output_i)`` where ``i=1..top_n``.

For the Config-API analogue of the pipeline below, please, check the
[config](https://github.com/OML-Team/open-metric-learning/blob/main/examples/sop/configs_experimental/train_postprocessor_sop.yaml).
The documentation for related classes is available via the [link](https://open-metric-learning.readthedocs.io/en/latest/contents/postprocessing.html).
*Note, this functionality is new and a work still in progress.*

<details>
<summary>Postprocessor: Training + Validation</summary>
<p>

[comment]:postprocessor-start

```python
from pprint import pprint

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from oml.datasets.base import DatasetWithLabels, DatasetQueryGallery
from oml.inference.flat import inference_on_dataframe
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.pairs import PairsMiner
from oml.models.siamese import ConcatSiamese
from oml.models.vit.vit import ViTExtractor
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.samplers.balance import BalanceSampler
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from oml.utils.download_mock_dataset import download_mock_dataset

# Let's start with saving embeddings of a pretrained extractor for which we want to build a postprocessor
dataset_root = "mock_dataset/"
download_mock_dataset(dataset_root)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)
transform = get_normalisation_resize_torch(im_size=64)

embeddings_train, embeddings_val, df_train, df_val = \
    inference_on_dataframe(dataset_root, "df.csv", extractor=extractor, transforms_extraction=transform)

# We are building Siamese model on top of existing weights and train it to recognize positive/negative pairs
siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])
optimizer = torch.optim.SGD(siamese.parameters(), lr=1e-6)
miner = PairsMiner(hard_mining=True)
criterion = BCEWithLogitsLoss()

train_dataset = DatasetWithLabels(df=df_train, transform=transform, extra_data={"embeddings": embeddings_train})
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)

for batch in train_loader:
    # We sample pairs on which the original model struggled most
    ids1, ids2, is_negative_pair = miner.sample(features=batch["embeddings"], labels=batch["labels"])
    probs = siamese(x1=batch["input_tensors"][ids1], x2=batch["input_tensors"][ids2])
    loss = criterion(probs, is_negative_pair.float())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Siamese re-ranks top-n retrieval outputs of the original model performing inference on pairs (query, output_i)
val_dataset = DatasetQueryGallery(df=df_val, extra_data={"embeddings": embeddings_val}, transform=transform)
valid_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

postprocessor = PairwiseImagesPostprocessor(top_n=3, pairwise_model=siamese, transforms=transform)
calculator = EmbeddingMetrics(postprocessor=postprocessor)
calculator.setup(num_samples=len(val_dataset))

for batch in valid_loader:
    calculator.update_data(data_dict=batch)

pprint(calculator.compute_metrics())  # Pairwise inference happens here
```
[comment]:postprocessor-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LBmusxwo8dPqWznmK627GNMzeDVdjMwv?usp=sharing)
