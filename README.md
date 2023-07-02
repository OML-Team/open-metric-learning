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

## [FAQ](https://open-metric-learning.readthedocs.io/en/latest/oml/faq.html)

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

* OML has [Pipelines](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines)
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
  widely used in the examples and custom `train` / `test` functions are used instead.

We believe that having Pipelines, laconic examples, and Zoo of pretrained models sets the entry threshold to a really low value.

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
<summary>How good may be a model trained with OML? </summary>
<p>

It may be comparable with the current (2022 year) [SotA](https://paperswithcode.com/task/metric-learning) methods,
for example, [Hyp-ViT](https://arxiv.org/pdf/2203.10833.pdf).
*(Few words about this approach: it's a ViT architecture trained with contrastive loss,
but the embeddings were projected into some hyperbolic space.
As the authors claimed, such a space is able to describe the nested structure of real-world data.
So, the paper requires some heavy math to adapt the usual operations for the hyperbolical space.)*

We trained the same architecture with triplet loss, fixing the rest of the parameters:
training and test transformations, image size, and optimizer. See configs in [Models Zoo](https://github.com/OML-Team/open-metric-learning#zoo).
The trick was in heuristics in our miner and sampler:

* [Category Balance Sampler](https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html#categorybalancesampler)
  forms the batches limiting the number of categories *C* in it.
  For instance, when *C = 1* it puts only jackets in one batch and only jeans into another one (just an example).
  It automatically makes the negative pairs harder: it's more meaningful for a model to realise why two jackets
  are different than to understand the same about a jacket and a t-shirt.

* [Hard Triplets Miner](https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html#hardtripletsminer)
  makes the task even harder keeping only the hardest triplets (with maximal positive and minimal negative distances).

Here are *CMC@1* scores for 2 popular benchmarks.
SOP dataset: Hyp-ViT — 85.9, ours — 86.6. DeepFashion dataset: Hyp-ViT — 92.5, ours — 92.1.
Thus, utilising simple heuristics and avoiding heavy math we are able to perform on SotA level.

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
we provide ready to use [Pipelines](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/).

The possibility of using pure PyTorch and modular structure of the code leaves a room for utilizing
OML with your favourite framework after the implementation of the necessary wrappers.

</p>
</details>


<details>
<summary>Can I use OML without any knowledge in DataScience?</summary>
<p>

Yes. To run the experiment with [Pipelines](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/)
you only need to write a converter
to our format (it means preparing the
`.csv` table with 5 predefined columns).
That's it!

Probably we already have a suitable pre-trained model for your domain
in our *Models Zoo*. In this case, you don't even need to train it.
</p>
</details>

## [Documentation](https://open-metric-learning.readthedocs.io/en/latest/index.html)

Documentation is available via the [link](https://open-metric-learning.readthedocs.io/en/latest/index.html).

You can also read some extra materials related to OML:

* Theory and practice of metric learning with the usage of OML.
[Post in English on Medium](https://medium.com/@AlekseiShabanov/practical-metric-learning-b0410cda2201) |
[Post in Russian on Habr](https://habr.com/ru/company/ods/blog/695380/) |
[Post in Chinese on CSDN](https://blog.csdn.net/fermion0217/article/details/127932087), translated by Chia-Chen Chang.

* The
[DEMO](https://dapladoc-oml-postprocessing-demo-srcappmain-pfh2g0.streamlit.app/)
for our paper
[STIR: Siamese Transformers for Image Retrieval Postprocessing](https://arxiv.org/abs/2304.13393)

* The report for Berlin-based meetup: "Computer Vision in production". November, 2022.
[Link](https://drive.google.com/drive/folders/1uHmLU8vMrMVMFodt36u0uXAgYjG_3D30?usp=share_link)

## [Installation](https://open-metric-learning.readthedocs.io/en/latest/oml/installation.html)

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

## [Examples](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#)

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
from oml.models import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
df_train, _ = download_mock_dataset(dataset_root)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).train()
optimizer = torch.optim.SGD(extractor.parameters(), lr=1e-6)

train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

for batch in tqdm(train_loader):
    embeddings = extractor(batch["input_tensors"])
    loss = criterion(embeddings, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # info for logging: positive/negative distances, number of active triplets
    print(criterion.last_logs)

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
from oml.models import ViTExtractor
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
_, df_val = download_mock_dataset(dataset_root)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).eval()

val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
calculator = EmbeddingMetrics(extra_keys=("paths",))
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        batch["embeddings"] = extractor(batch["input_tensors"])
        calculator.update_data(batch)

metrics = calculator.compute_metrics()

# Logging
print(calculator.metrics)  # metrics
print(calculator.metrics_unreduced)  # metrics without averaging over queries

# Visualisation
calculator.get_plot_for_queries(query_ids=[0, 2], n_instances=5)  # draw predictions on predefined queries
calculator.get_plot_for_worst_queries(metric_name="OVERALL/map/5", n_queries=2, n_instances=5)  # draw mistakes
calculator.visualize()  # draw mistakes for all the available metrics

```
[comment]:vanilla-validation-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O2o3k8I8jN5hRin3dKnAS3WsgG04tmIT?usp=sharing)
<details>
<summary>Training + Validation [Lightning and Neptune logging]</summary>
<p>

[comment]:lightning-start
```python
import pytorch_lightning as pl
import torch

from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.modules.extractor import ExtractorModule
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

dataset_root = "mock_dataset/"
df_train, df_val = download_mock_dataset(dataset_root)

# model
extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False)

# train
optimizer = torch.optim.SGD(extractor.parameters(), lr=1e-6)
train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=3)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)

# val
val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
metric_callback = MetricValCallback(metric=EmbeddingMetrics(extra_keys=[train_dataset.paths_key,]), log_images=True)

# logging
logger = TensorBoardLogger(".")  # For TensorBoard
# logger = NeptuneLogger(api_key="", project="", log_model_checkpoints=False)  # For Neptune

# run
pl_model = ExtractorModule(extractor, criterion, optimizer)
trainer = pl.Trainer(max_epochs=3, callbacks=[metric_callback], num_sanity_val_steps=0, logger=logger)
trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

```
[comment]:lightning-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bVUgdBGWvQgCkba2YtaIRVlUQUz7Q60Z?usp=share_link)
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

extractor = ViTExtractor.from_pretrained("vits16_dino")
transform, _ = get_transforms_for_pretrained("vits16_dino")

args = {"num_workers": 0, "batch_size": 8}
features_queries = inference_on_images(extractor, paths=queries, transform=transform, **args)
features_galleries = inference_on_images(extractor, paths=galleries, transform=transform, **args)

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

[**Schemas, explanations and tips**](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction)

See [extra code snippets](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html), including:
* Training + Validation with Lightning
* Training + Validation with Lightning in DDP mode
* Training with losses from PML
* Training with losses from PML advanced (passing distance, reducer, miner)

## [Pipelines](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines)

Pipelines provide a way to run metric learning experiments via changing only the config file.
All you need is to prepare your dataset in a required format.

See [Pipelines](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/) folder for more details:
* Feature extractor [pipeline](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction)
* Retrieval postprocessor [pipeline](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing) (re-ranking)

## [Zoo](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/zoo.html)

Below are the models trained with OML on 4 public datasets.
All metrics below were obtained on the images with the sizes of **224 x 224**:

|                      model                      | cmc1  |         dataset          |                                              weights                                              |                                                    experiment                                                     |
|:-----------------------------------------------:|:-----:|:------------------------:|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| `ViTExtractor.from_pretrained("vits16_inshop")` | 0.921 |    DeepFashion Inshop    |    [link](https://drive.google.com/file/d/1niX-TC8cj6j369t7iU2baHQSVN3MVJbW/view?usp=sharing)     | [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_inshop) |
|  `ViTExtractor.from_pretrained("vits16_sop")`   | 0.866 | Stanford Online Products |   [link](https://drive.google.com/file/d/1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A/view?usp=sharing)      |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_sop)   |
| `ViTExtractor.from_pretrained("vits16_cars")`   | 0.907 |         CARS 196         |   [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_cars)  |
|  `ViTExtractor.from_pretrained("vits16_cub")`   | 0.837 |       CUB 200 2011       |   [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_cub)   |

We also provide an integration with the models pretrained by other researchers.
All metrics below were obtained on the images with the sizes of **224 x 224**:

|                            model                             | Stanford Online Products | DeepFashion InShop | CUB 200 2011 | CARS 196 |
|:------------------------------------------------------------:|:------------------------:|:------------------:|:------------:|:--------:|
|    `ViTUnicomExtractor.from_pretrained("vitb16_unicom")`     |          0.700           |       0.734        |    0.847     |  0.916   |
|    `ViTUnicomExtractor.from_pretrained("vitb32_unicom")`     |          0.690           |       0.722        |    0.796     |  0.893   |
|    `ViTUnicomExtractor.from_pretrained("vitl14_unicom")`     |          0.726           |       0.790        |    0.868     |  0.922   |
| `ViTUnicomExtractor.from_pretrained("vitl14_336px_unicom")`  |          0.745           |       0.810        |    0.875     |  0.924   |
|    `ViTCLIPExtractor.from_pretrained("sber_vitb32_224")`     |          0.547           |       0.514        |    0.448     |  0.618   |
|    `ViTCLIPExtractor.from_pretrained("sber_vitb16_224")`     |          0.565           |       0.565        |    0.524     |  0.648   |
|    `ViTCLIPExtractor.from_pretrained("sber_vitl14_224")`     |          0.512           |       0.555        |    0.606     |  0.707   |
|   `ViTCLIPExtractor.from_pretrained("openai_vitb32_224")`    |          0.612           |       0.491        |    0.560     |  0.693   |
|   `ViTCLIPExtractor.from_pretrained("openai_vitb16_224")`    |          0.648           |       0.606        |    0.665     |  0.767   |
|   `ViTCLIPExtractor.from_pretrained("openai_vitl14_224")`    |          0.670           |       0.675        |    0.745     |  0.844   |
|        `ViTExtractor.from_pretrained("vits16_dino")`         |          0.648           |       0.509        |    0.627     |  0.265   |
|         `ViTExtractor.from_pretrained("vits8_dino")`         |          0.651           |       0.524        |    0.661     |  0.315   |
|        `ViTExtractor.from_pretrained("vitb16_dino")`         |          0.658           |       0.514        |    0.541     |  0.288   |
|         `ViTExtractor.from_pretrained("vitb8_dino")`         |          0.689           |       0.599        |    0.506     |  0.313   |
|    `ResnetExtractor.from_pretrained("resnet50_moco_v2")`     |          0.493           |       0.267        |    0.264     |  0.149   |
| `ResnetExtractor.from_pretrained("resnet50_imagenet1k_v1")`  |          0.515           |       0.284        |    0.455     |  0.247   |

**The metrics may be different from the ones reported by papers,
because the version of train/val split and usage of bounding boxes may differ.
Particularly, we used bounding boxes during the evaluation.*

### How to use models from Zoo?

[comment]:zoo-start
```python
from oml.const import CKPT_SAVE_ROOT as CKPT_DIR, MOCK_DATASET_PATH as DATA_DIR
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained

model = ViTExtractor.from_pretrained("vits16_dino")
transforms, im_reader = get_transforms_for_pretrained("vits16_dino")

img = im_reader(DATA_DIR / "images" / "circle_1.jpg")  # put path to your image here
img_tensor = transforms(img)
# img_tensor = transforms(image=img)["image"]  # for transforms from Albumentations

features = model(img_tensor.unsqueeze(0))

# Check other available models:
print(list(ViTExtractor.pretrained_models.keys()))

# Load checkpoint saved on a disk:
model_ = ViTExtractor(weights=CKPT_DIR / "vits16_dino.ckpt", arch="vits16", normalise_features=False)
```
[comment]:zoo-end

## [Contributing guide](https://open-metric-learning.readthedocs.io/en/latest/oml/contributing.html)

We welcome new contributors! Please, see our:
* [Contributing guide](https://open-metric-learning.readthedocs.io/en/latest/from_readme/contributing.html)
* [Kanban board](https://github.com/OML-Team/open-metric-learning/projects/1)

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
