<div align="center">
<img src="https://i.ibb.co/wsmD5r4/photo-2022-06-06-17-40-52.jpg" width="400px">

![example workflow](https://github.com/OML-Team/open-metric-learning/actions/workflows/pre-commit-workflow.yaml/badge.svg)
![example workflow](https://github.com/OML-Team/open-metric-learning/actions/workflows/tests-workflow.yaml/badge.svg?)
[![Documentation Status](https://readthedocs.org/projects/open-metric-learning/badge/?version=latest)](https://open-metric-learning.readthedocs.io/en/latest/?badge=latest)
[![PyPI Status](https://pepy.tech/badge/open-metric-learning)](https://pepy.tech/project/open-metric-learning)
[![Pipi version](https://img.shields.io/pypi/v/open-metric-learning.svg)](https://pypi.org/project/open-metric-learning/)
![example workflow](https://github.com/OML-Team/open-metric-learning/actions/workflows/python-versions.yaml/badge.svg?)
[![python](https://img.shields.io/badge/python_3.7-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/python-versions.yaml/badge.svg?)
[![python](https://img.shields.io/badge/python_3.8-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/python-versions.yaml/badge.svg?)
[![python](https://img.shields.io/badge/python_3.9-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/python-versions.yaml/badge.svg?)
[![python](https://img.shields.io/badge/python_3.10-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/python-versions.yaml/badge.svg?)

<div align="left">

OML is a PyTorch-based framework to train and validate the models producing high-quality embeddings.

## FAQ

<details>
<summary>Why do I need OML?</summary>
<p>

You may think *"If I need image embeddings I can simply train a vanilla classifier and take its penultimate layer"*.
Well, it makes sense as a starting point. But there are several possible drawbacks:

* If you want to use embeddings to perform searching you need to calculate some distance among them (for example, cosine or L2).
  Usually, **you don't directly optimize these distances during the training** in the classification setup. So, you can only hope that
  final embeddings will have the desired properties.

* **The second problem is the validation process**.
  In the searching setup, you usually care how related your top-N outputs are to the query.
  The natural way to evaluate the model is to simulate searching requests to the reference set
  and apply one of the retrieval metrics.
  So, there is no guarantee that classification accuracy will correlate with these metrics.

* Finally, you may want to implement a metric learning pipeline by yourself.
  **There is a lot of work**: to use triplet loss you need to form batches in a specific way,
  implement different kinds of triplets mining, tracking distances, etc. For the validation, you also need to
  implement retrieval metrics,
  which include effective embeddings accumulation during the epoch, covering corner cases, etc.
  It's even harder if you have several gpus and use DDP.
  You may also want to visualize your search requests by highlighting good and bad search results.
  Instead of doing it by yourself, you can simply use OML for your purposes.

</p>
</details>


<details>
<summary>What is the difference between Open Metric Learning and PyTorch Metric Learning?</summary>
<p>

[PML](https://github.com/KevinMusgrave/pytorch-metric-learning) is the popular library for Metric Learning,
and it includes a rich collection of losses, miners, distances, and reducers; that is why we provide straightforward
[examples](https://github.com/OML-Team/open-metric-learning#usage-with-pytorch-metric-learning) of using them with OML.
Initially, we tried to use PML, but in the end, we came up with our library, which is more pipeline / recipes oriented.
That is how OML differs from PML:

* OML has [Config API](https://open-metric-learning.readthedocs.io/en/latest/examples/config.html)
  which allows training models by preparing a config and your data in the required format
  (it's like converting data into COCO format to train a detector from [mmdetection](https://github.com/open-mmlab/mmdetection)).

* OML focuses on end-to-end pipelines and practical use cases.
  It has config based examples on popular benchmarks close to real life (like photos of products of thousands ids).
  We found some good combinations of hyperparameters on these datasets, trained and published models and their configs.
  Thus, it makes OML more recipes oriented than PML, and its author
  [confirms](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/169#issuecomment-670814393)
  this saying that his library is a set of tools rather the recipes, moreover, the examples in PML are mostly for CIFAR and MNIST datasets.

* OML has the [Zoo](https://github.com/OML-Team/open-metric-learning#zoo) of pretrained models that can be easily accessed from
  the code in the same way as in `torchvision` (when you type `resnet50(pretrained=True)`).

* OML is integrated with [PyTorch Lightning](https://www.pytorchlightning.ai/), so, we can use the power of its
  [Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).
  This is especially helpful when we work with DDP, so, you compare our
  [DDP example](https://open-metric-learning.readthedocs.io/en/latest/examples/python.html)
  and the
  [PMLs one](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/DistributedTripletMarginLossMNIST.ipynb).
  By the way, PML also has [Trainers](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/), but it's not
  in the examples and custom `train` / `test` functions are used instead.

We believe that having Config API, laconic examples, and Zoo of pretrained models sets the entry threshold to a really low value.

</p>
</details>


<details>
<summary>What is Metric Learning?</summary>
<p>

Metric Learning problem (also known as *extreme classification* problem) means a situation in which we
have thousands of ids of some entities, but only a few samples for every entity.
Often we assume that during the test stage (or production) we will deal with unseen entities
which makes it impossible to apply the vanilla classification pipeline directly. In many cases obtained embeddings
are used to perform search or matching procedures over them.

Here are a few examples of such tasks from the computer vision sphere:
* Person/Animal Re-Identification
* Face Recognition
* Landmark Recognition
* Searching engines for online shops
 and many others.
</p>
</details>


<details>
<summary>Glossary (Naming convention) </summary>
<p>

* `embedding` - model's output (also known as `features vector` or `descriptor`).
* `query` - a sample which is used as a request in the retrieval procedure.
* `gallery set` - the set of entities to search items similar to `query` (also known as `reference` or `index`).
* `Sampler` - an argument for `DataLoader` which is used to form batches
* `Miner` - the object to form pairs or triplets after the batch was formed by `Sampler`. It's not necessary to form
  the combinations of samples only inside the current batch, thus, the memory bank may be a part of `Miner`.
* `Samples`/`Labels`/`Instances` - as an example let's consider DeepFashion dataset. It includes thousands of
  fashion item ids (we name them `labels`) and several photos for each item id
  (we name the individual photo as `instance` or `sample`). All of the fashion item ids have their groups like
  "skirts", "jackets", "shorts" and so on (we name them `categories`).
  Note, we avoid using the term `class` to avoid misunderstanding.
* `training epoch` - batch samplers which we use for combination-based losses usually have a length equal to
  `[number of labels in training dataset] / [numbers of labels in one batch]`. It means that we don't observe all of
  the available training samples in one epoch (as opposed to vanilla classification),
  instead, we observe all of the available labels.

</p>
</details>


<details>
<summary>How does OML work under the hood? </summary>
<p>

**Training part** implies using losses, well-established for metric learning, such as the angular losses
(like *ArcFace*) or the combinations based losses (like *TripletLoss* or *ContrastiveLoss*).
The latter benefits from effective mining schemas of triplets/pairs, so we pay great attention to it.
Thus, during the training we:
   1. Use `DataLoader` + `Sampler` to form batches (for example `BalanceSampler`)
   2. [Only for losses based on combinations] Use `Miner` to form effective pairs or triplets, including those which utilize a memory bank.
   3. Compute loss.

**Validation part** consists of several steps:
  1. Accumulating all of the embeddings (`EmbeddingMetrics`).
  2. Calculating distances between them with respect to query/gallery split.
  3. Applying some specific retrieval techniques like query reranking or score normalisation.
  4. Calculating retrieval metrics like *CMC@k*, *Precision@k* or *MeanAveragePrecision@k*.

</p>
</details>


<details>
<summary>What about Self-Supervised Learning?</summary>
<p>

Recent research in SSL definitely obtained great results. The problem is that these approaches
required an enormous amount of computing to train the model. But in our framework, we consider the most common case
when the average user has no more than a few GPUs.

At the same time, it would be unwise to ignore success in this sphere, so we still exploit it in two ways:
* As a source of checkpoints that would be great to start training with. From publications and our experience,
  they are much better as initialisation than the default supervised model trained on ImageNet. Thus, we added the possibility
  to initialise your models using these pretrained checkpoints only by passing an argument in the config or the constructor.
* As a source of inspiration. For example, we adapted the idea of a memory bank from *MoCo* for the *TripletLoss*.

</p>
</details>


<details>
<summary>Do I need to know other frameworks to use OML?</summary>
<p>

No, you don't. OML is a framework-agnostic. Despite we use PyTorch Lightning as a loop
runner for the experiments, we also keep the possibility to run everything on pure PyTorch.
Thus, only the tiny part of OML is Lightning-specific and we keep this logic separately from
other code (see `oml.lightning`). Even when you use Lightning, you don't need to know it, since
we provide ready to use [Config API](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).

The possibility of using pure PyTorch and modular structure of the code leaves a room for utilizing
OML with your favourite framework after the implementation of the necessary wrappers.

</p>
</details>


<details>
<summary>Can I use OML without any knowledge in DataScience?</summary>
<p>

Yes. To run the experiment with [Config API](https://github.com/OML-Team/open-metric-learning/blob/main/examples/)
you only need to write a converter
to our format (it means preparing the
`.csv` table with 5 predefined columns).
That's it!

Probably we already have a suitable pre-trained model for your domain
in our *Models Zoo*. In this case, you don't even need to train it.
</p>
</details>

## Documentation

Documentation is available via the [link](https://open-metric-learning.readthedocs.io/en/latest/index.html).

## Installation

OML is available in PyPI:

```shell
pip install -U open-metric-learning
```

You can also pull the prepared image from DockerHub...

```shell
docker pull omlteam/oml:gpu
docker pull omlteam/oml:cpu
```

...or build one by your own

```shell
make docker_build RUNTIME=cpu
make docker_build RUNTIME=gpu
```

## Get started using Config API

Using configs is the best option if your dataset and pipeline are standard enough or if you are not
experienced in Machine Learning or Python. You can find more details in the
[examples](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).

## Get started using Python

The most flexible, but knowledge-requiring approach.
You are not limited by our project structure and you can use only that part of the functionality which you need.
You can start with fully working code snippets below that train and validate the model
on a tiny dataset of
[figures](https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=sharing).


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
ㅤ
ㅤ

If you want to train your model in the DDP regime (Distributed Data Parallel), you
only need to slightly change only few lines of code in the example below.

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


## Usage with PyTorch Metric Learning

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

## Zoo

Below are the models trained with OML on 4 public datasets.
For more details about the training process, please, visit *examples* submodule and it's
[Readme](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).

|                            model                            | cmc1  |         dataset          |                                              weights                                              |                                           configs                                            | hash (the beginning) |
|:-----------------------------------------------------------:|:-----:|:------------------------:|:-------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------:|
| `ViTExtractor(weights="vits16_inshop", arch="vits16", ...)` | 0.921 |    DeepFashion Inshop    |    [link](https://drive.google.com/file/d/1niX-TC8cj6j369t7iU2baHQSVN3MVJbW/view?usp=sharing)     | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/inshop/configs)   |        e1017d        |
|  `ViTExtractor(weights="vits16_sop", arch="vits16", ...)`   | 0.866 | Stanford Online Products |   [link](https://drive.google.com/file/d/1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A/view?usp=sharing)      | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/sop/configs)      |        85cfa5        |
|  `ViTExtractor(weights="vits16_cars", arch="vits16", ...)`  | 0.907 |         CARS 196         |   [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing)    | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cars/configs)     |        9f1e59        |
|  `ViTExtractor(weights="vits16_cub", arch="vits16", ...)`   | 0.837 |       CUB 200 2011       |   [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing)    | [link](https://github.com/OML-Team/open-metric-learning/tree/main/examples/cub/configs)      |        e82633        |

We also provide an integration with the models pretrained by other researchers:

|                        model                              |   Stanford Online Products |   DeepFashion InShop |   CUB 200 2011 |   CARS 196 |
|:---------------------------------------------------------:|:--------------------------:|:--------------------:|:--------------:|:----------:|
| `ViTCLIPExtractor("sber_vitb32_224", "vitb32_224")`       |                      0.547 |                0.514 |          0.448 |      0.618 |
| `ViTCLIPExtractor("sber_vitb16_224", "vitb16_224")`       |                      0.565 |                0.565 |          0.524 |      0.648 |
| `ViTCLIPExtractor("sber_vitl14_224", "vitl14_224")`       |                      0.512 |                0.555 |          0.606 |      0.707 |
| `ViTCLIPExtractor("openai_vitb32_224", "vitb32_224")`     |                      0.612 |                0.491 |          0.560 |      0.693 |
| `ViTCLIPExtractor("openai_vitb16_224", "vitb16_224")`     |                      0.648 |                0.606 |          0.665 |      0.767 |
| `ViTCLIPExtractor("openai_vitl14_224", "vitl14_224")`     |                      0.670 |                0.675 |          0.745 |      0.844 |
| `ViTExtractor("vits16_dino", "vits16")`                   |                      0.629 |                0.456 |          0.693 |      0.313 |
| `ViTExtractor("vits8_dino", "vits8")`                     |                      0.637 |                0.478 |          0.703 |      0.344 |
| `ViTExtractor("vitb16_dino", "vitb16")`                   |                      0.636 |                0.464 |          0.626 |      0.340 |
| `ViTExtractor("vitb8_dino", "vitb8")`                     |                      0.673 |                0.548 |          0.546 |      0.342 |
| `ResnetExtractor("resnet50_moco_v2", "resnet50")`         |                      0.491 |                0.310 |          0.244 |      0.155 |

*All figures above were obtained on the images with the sizes of 224 x 224.
Note, that the models above expect the crop of the region of interest rather than the whole picture.
It is also important to say that different models expect different preprocessing.
You should use `norm_resize_albu_clip` for `ViTCLIPExtractor` and `norm_resize_albu` for all other models
(note that you can find this transforms in `oml.registry.transforms.TRANSFORMS_REGISTRY`).*


You can specify the desired weights and architecture to automatically download pretrained checkpoint (by the analogue with `torchvision.models`):

[comment]:checkpoint-start
```python
import oml
from oml.models.vit.vit import ViTExtractor
from oml.registry.models import MODELS_REGISTRY

# We are downloading vits16 pretrained on CARS dataset:
model = ViTExtractor(weights="vits16_cars", arch="vits16", normalise_features=False)

# You can also check other available pretrained models...
print(list(ViTExtractor.pretrained_models.keys()))

# ...or check other available types of architectures
print(MODELS_REGISTRY)

# It's also possible to use `weights` argument to directly pass the path to the checkpoint:
model_from_disk = ViTExtractor(weights=oml.const.CKPT_SAVE_ROOT / "vits16_cars.ckpt", arch="vits16", normalise_features=False)
```
[comment]:checkpoint-end

## Contributing guide

We welcome new contributors! Please, see our
[contributing guide](https://open-metric-learning.readthedocs.io/en/latest/from_readme/contributing.html).

## Acknowledgments

<a href="https://github.com/catalyst-team/catalyst" target="_blank"><img src="https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png" width="100"/></a>

The project was started in 2020 as a module for [Catalyst](https://github.com/catalyst-team/catalyst) library.
I want to thank people who worked with me on that module:
[Julia Shenshina](https://github.com/julia-shenshina),
[Nikita Balagansky](https://github.com/elephantmipt),
[Sergey Kolesnikov](https://github.com/Scitator)
and others.

I would like to thank people who continue working on this pipeline when it became a separe project:
[Julia Shenshina](https://github.com/julia-shenshina),
[Misha Kindulov](https://github.com/b0nce),
[Aleksei Tarasov](https://github.com/DaloroAT) and
[Verkhovtsev Leonid](https://github.com/leoromanovich).

<a href="https://www.newyorker.de/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/New_Yorker.svg/1280px-New_Yorker.svg.png" width="100"/></a>

I also want to thank NewYorker, since the part of functionality was developed (and used) by its computer vision team led by me.
