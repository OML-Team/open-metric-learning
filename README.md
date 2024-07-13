<div align="center">
<img src="https://i.ibb.co/wsmD5r4/photo-2022-06-06-17-40-52.jpg" width="400px">


[![Documentation Status](https://readthedocs.org/projects/open-metric-learning/badge/?version=latest)](https://open-metric-learning.readthedocs.io/en/latest/?badge=latest)
[![PyPI Status](https://pepy.tech/badge/open-metric-learning)](https://pepy.tech/project/open-metric-learning)
[![Pipi version](https://img.shields.io/pypi/v/open-metric-learning.svg)](https://pypi.org/project/open-metric-learning/)
[![python](https://img.shields.io/badge/python_3.8-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/tests.yaml/badge.svg?)
[![python](https://img.shields.io/badge/python_3.9-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/tests.yaml/badge.svg?)
[![python](https://img.shields.io/badge/python_3.10-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/tests.yaml/badge.svg?)
[![python](https://img.shields.io/badge/python_3.11-passing-success)](https://github.com/OML-Team/open-metric-learning/actions/workflows/tests.yaml/badge.svg?)


OML is a PyTorch-based framework to train and validate the models producing high-quality embeddings.

### Trusted by

<div align="center">
<a href="https://docs.neptune.ai/integrations/community_developed/" target="_blank"><img src="https://security.neptune.ai/api/share/b707f1e8-e287-4f01-b590-39a6fa7e9faa/logo.png" width="100"/></a>ㅤㅤ
<a href="https://www.newyorker.de/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/New_Yorker.svg/1280px-New_Yorker.svg.png" width="100"/></a>ㅤㅤ
<a href="https://www.epoch8.co/" target="_blank"><img src="https://i.ibb.co/GdNVTyt/Screenshot-2023-07-04-at-11-19-24.png" width="100"/></a>ㅤㅤ
<a href="https://www.meituan.com" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/6/61/Meituan_English_Logo.png" width="100"/></a>ㅤㅤ
<a href="https://constructor.io/" target="_blank"><img src="https://rethink.industries/wp-content/uploads/2022/04/constructor.io-logo.png" width="100"/></a>ㅤㅤ
<a href="https://edgify.ai/" target="_blank"><img src="https://edgify.ai/wp-content/uploads/2024/04/new-edgify-logo.svg" width="100" height="30"/></a>ㅤㅤ
<a href="https://inspector-cloud.ru/" target="_blank"><img src="https://thumb.tildacdn.com/tild6533-6433-4137-a266-613963373637/-/resize/540x/-/format/webp/photo.png" width="150" height="30"/></a>ㅤㅤ
<a href="https://yango-tech.com/" target="_blank"><img src="https://yango-backend.sborkademo.com/media/pages/home/205f66f309-1717169752/opengr4-1200x630-crop-q85.jpg" width="100" height="30"/></a>ㅤㅤ
<a href="https://www.adagrad.ai/" target="_blank"><img src="https://assets-global.website-files.com/619cafd224a31d1835ece5bd/61de7f23546e9662e51605ba_Adagrad_logo_footer-2022.png" width="100" height="30"/></a>

<a href="https://www.ox.ac.uk/" target="_blank"><img src="https://i.ibb.co/zhWL6tD/21-05-2019-16-08-10-6922268.png" width="120"/></a>ㅤㅤ
<a href="https://www.hse.ru/en/" target="_blank"><img src="https://www.hse.ru/data/2020/11/16/1367274044/HSE_University_blue.jpg.(230x86x123).jpg" width="100"/></a>

There is a number of people from
[Oxford](https://www.ox.ac.uk/) and
[HSE](https://www.hse.ru/en/)
universities who have used OML in their theses.
[[1]](https://github.com/nilomr/open-metric-learning/tree/great-tit/great-tit-train)
[[2]](https://github.com/nastygorodi/PROJECT-Deep_Metric_Learning)
[[3]](https://github.com/nik-fedorov/term_paper_metric_learning)


<div align="left">


<details>
<summary><b>OML 3.0 has been released!</b></summary>
<p>

The update focuses on several components:

* We added "official" texts support and the corresponding Python examples. (Note, texts support in Pipelines is not supported yet.)

* We introduced the `RetrievalResults` (`RR`) class — a container to store gallery items retrieved for given queries.
`RR` provides a unified way to visualize predictions and compute metrics (if ground truths are known).
It also simplifies post-processing, where an `RR` object is taken as input and another `RR_upd` is produced as output.
Having these two objects allows comparison retrieval results visually or by metrics.
Moreover, you can easily create a chain of such post-processors.
  * `RR` is memory optimized because of using batching: in other words, it doesn't store full matrix of query-gallery distances.
    (It doesn't make search approximate though).

* We made `Model` and `Dataset` the only classes responsible for processing modality-specific logic.
`Model` is responsible for interpreting its input dimensions: for example, `BxCxHxW` for images or `BxLxD` for sequences like texts.
`Dataset` is responsible for preparing an item: it may use `Transforms` for images or `Tokenizer` for texts.
Functions computing metrics like `calc_retrieval_metrics_rr`, `RetrievalResults`, `PairwiseReranker`, and other classes and functions are unified
to work with any modality.
  * We added `IVisualizableDataset` having method `.visaulize()` that shows a single item. If implemented,
   `RetrievalResults` is able to show the layout of retrieved results.

#### Migration from OML 2.* [Python API]:

The easiest way to catch up with changes is to re-read the examples!

* The recommended way of validation is to use `RetrievalResults` and functions like `calc_retrieval_metrics_rr`,
`calc_fnmr_at_fmr_rr`, and others. The `EmbeddingMetrics` class is kept for use with PyTorch Lightning and inside Pipelines.
Note, the signatures of `EmbeddingMetrics` methods have been slightly changed, see Lightning examples for that.

* Since modality-specific logic is confined to `Dataset`, it doesn't output `PATHS_KEY`, `X1_KEY`, `X2_KEY`, `Y1_KEY`, and `Y2_KEY` anymore.
Keys which are not modality-specific like `LABELS_KEY`, `IS_GALLERY`, `IS_QUERY_KEY`, `CATEGORIES_KEY` are still in use.

* `inference_on_images` is now `inference` and works with any modality.

* Slightly changed interfaces of `Datasets.` For example, we have `IQueryGalleryDataset` and `IQueryGalleryLabeledDataset` interfaces.
  The first has to be used for inference, the second one for validation. Also added `IVisualizableDataset` interface.

* Removed some internals like `IMetricDDP`, `EmbeddingMetricsDDP`, `calc_distance_matrix`, `calc_gt_mask`, `calc_mask_to_ignore`,
  `apply_mask_to_ignore`. These changes shouldn't affect you. Also removed code related to a pipeline with precomputed triplets.

#### Migration from OML 2.* [Pipelines]:

* [Feature extraction](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction):
No changes, except for adding an optional argument — `mode_for_checkpointing = (min | max)`. It may be useful
to switch between *the lower, the better* and *the greater, the better* type of metrics.

* [Pairwise-postprocessing pipeline](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing):
Slightly changed the name and arguments of the `postprocessor` sub config — `pairwise_images` is now `pairwise_reranker`
and doesn't need transforms.

</p>
</details>

## [Documentation](https://open-metric-learning.readthedocs.io/en/latest/index.html)

<details>
<summary>FAQ</summary>

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
[examples](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-metric-learning) of using them with OML.
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
  [DDP example](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning)
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
`.csv` table with a few predefined columns).
That's it!

Probably we already have a suitable pre-trained model for your domain
in our *Models Zoo*. In this case, you don't even need to train it.
</p>
</details>


<details>
<summary>Can I export models to ONNX?</summary>
<p>

Currently, we don't support exporting models to ONNX directly.
However, you can use the built-in PyTorch capabilities to achieve this. For more information, please refer to this [issue](https://github.com/OML-Team/open-metric-learning/issues/592).
</p>
</details>

</details>


[DOCUMENTATION](https://open-metric-learning.readthedocs.io/en/latest/index.html)

TUTORIAL TO START WITH:
[English](https://medium.com/@AlekseiShabanov/practical-metric-learning-b0410cda2201) |
[Russian](https://habr.com/ru/company/ods/blog/695380/) |
[Chinese](https://zhuanlan.zhihu.com/p/683102241)

<details>
<summary>MORE</summary>

* The
[DEMO](https://dapladoc-oml-postprocessing-demo-srcappmain-pfh2g0.streamlit.app/)
for our paper
[STIR: Siamese Transformers for Image Retrieval Postprocessing](https://arxiv.org/abs/2304.13393)

* Meet OpenMetricLearning (OML) on
[Marktechpost](https://www.marktechpost.com/2023/12/26/meet-openmetriclearning-oml-a-pytorch-based-python-framework-to-train-and-validate-the-deep-learning-models-producing-high-quality-embeddings/)

* The report for Berlin-based meetup: "Computer Vision in production". November, 2022.
[Link](https://drive.google.com/drive/folders/1uHmLU8vMrMVMFodt36u0uXAgYjG_3D30?usp=share_link)

</details>

## [Installation](https://open-metric-learning.readthedocs.io/en/latest/oml/installation.html)

```shell
pip install -U open-metric-learning; # minimum dependencies
pip install -U open-metric-learning[nlp]
pip install -U open-metric-learning[audio]
```

<details><summary>DockerHub</summary>

```shell
docker pull omlteam/oml:gpu
docker pull omlteam/oml:cpu
```

</details>


## OML features

<div style="overflow-x: auto;">

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html"> <b>Losses</b></a> |
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html"> <b>Miners</b></a>

```python
miner = AllTripletsMiner()
miner = NHardTripletsMiner()
miner = MinerWithBank()
...
criterion = TripletLossWithMiner(0.1, miner)
criterion = ArcFaceLoss()
criterion = SurrogatePrecision()
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html"> <b>Samplers</b></a>

```python
labels = train.get_labels()
l2c = train.get_label2category()


sampler = BalanceSampler(labels)
sampler = CategoryBalanceSampler(labels, l2c)
sampler = DistinctCategoryBalanceSampler(labels, l2c)
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/"><b>Configs support</b></a>

```yaml
max_epochs: 10
sampler:
  name: balance
  args:
    n_labels: 2
    n_instances: 2
```

</td>
<td style="text-align: left;">
<a href="https://github.com/OML-Team/open-metric-learning?tab=readme-ov-file#zoo"><b>Pre-trained models</b></a>

```python
model_hf = AutoModel.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
extractor_txt = HFWrapper(model_hf)

extractor_img = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/algo_examples.html"><b>Post-processing</b></a>

```python
emb = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(emb, dataset)

postprocessor = AdaptiveThresholding()
rr_upd = postprocessor.process(rr, dataset)
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/postprocessing/siamese_examples.html"><b>Post-processing by NN</b></a> |
<a href="https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing"><b>Paper</b></a>

```python
embeddings = inference(extractor, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

postprocessor = PairwiseReranker(ConcatSiamese(), top_n=3)
rr_upd = postprocessor.process(rr, dataset)
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/oml/logging.html#"><b>Logging</b></a><br>

```python
logger = TensorBoardPipelineLogger()
logger = NeptunePipelineLogger()
logger = WandBPipelineLogger()
logger = MLFlowPipelineLogger()
logger = ClearMLPipelineLogger()
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-metric-learning"><b>PML</b></a><br>

```python
from pytorch_metric_learning import losses

criterion = losses.TripletMarginLoss(0.2, "all")
pred = ViTExtractor()(data)
criterion(pred, gts)
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#handling-categories"><b>Categories support</b></a>

```python
# train
loader = DataLoader(CategoryBalanceSampler())

# validation
rr = RetrievalResults.from_embeddings()
m.calc_retrieval_metrics_rr(rr, query_categories)
```

</td>
<td style="text-align: left;"><a href="https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html"><b>Misc metrics</b></a>

```python
embeddigs = inference(model, dataset)
rr = RetrievalResults.from_embeddings(embeddings, dataset)

m.calc_retrieval_metrics_rr(rr, precision_top_k=(5,))
m.calc_fnmr_at_fmr_rr(rr, fmr_vals=(0.1,))
m.calc_topological_metrics(embeddings, pcf_variance=(0.5,))
```

</td>
</tr>

<tr>
</tr>

<tr>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning"><b>Lightning</b></a><br>

```python
import pytorch_lightning as pl

model = ViTExtractor.from_pretrained("vits16_dino")
clb = MetricValCallback(EmbeddingMetrics(dataset))
module = ExtractorModule(model, criterion, optimizer)

trainer = pl.Trainer(max_epochs=3, callbacks=[clb])
trainer.fit(module, train_loader, val_loader)
```

</td>
<td style="text-align: left;">
<a href="https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#usage-with-pytorch-lightning"><b>Lightning DDP</b></a><br>

```python
clb = MetricValCallback(EmbeddingMetrics(val))
module = ExtractorModuleDDP(
    model, criterion, optimizer, train, val
)

ddp = {"devices": 2, "strategy": DDPStrategy()}
trainer = pl.Trainer(max_epochs=3, callbacks=[clb], **ddp)
trainer.fit(module)
```

</td>
</tr>

</table>

</div>

## [Examples](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html#)

Here is an example of how to train, validate and post-process the model
on a tiny dataset of
[images](https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4)
or
[texts](https://github.com/OML-Team/open-metric-learning/blob/main/oml/utils/download_mock_dataset.py#L83).
See more details on dataset
[format](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).

<div style="overflow-x: auto;">

<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">

<tr>
</tr>

<tr>
    <td style="text-align: left; padding: 0;"><b>IMAGES</b></td>
    <td style="text-align: left; padding: 0;"><b>TEXTS</b></td>
</tr>

<tr>
</tr>

<tr>

<td>

[comment]:train-val-img-start
```python
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset

model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").train()
transform, _ = get_transforms_for_pretrained("vits16_dino")

df_train, df_val = get_mock_images_dataset(global_paths=True)
train = d.ImageLabeledDataset(df_train, transform=transform)
val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)


def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr = AdaptiveThresholding(n_std=2).process(rr)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)
    print(calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()
```
[comment]:train-val-img-end

</td>

<td>

[comment]:train-val-txt-start
```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import HFWrapper
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler
from oml.utils import get_mock_texts_dataset

model = HFWrapper(AutoModel.from_pretrained("bert-base-uncased"), 768).to("cpu").train()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

df_train, df_val = get_mock_texts_dataset()
train = d.TextLabeledDataset(df_train, tokenizer=tokenizer)
val = d.TextQueryGalleryLabeledDataset(df_val, tokenizer=tokenizer)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)


def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr = AdaptiveThresholding(n_std=2).process(rr)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)
    print(calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()
```
[comment]:train-val-txt-end
</td>
</tr>

<tr>
</tr>

<tr>

<td>

<details style="padding-bottom: 10px">
<summary>Output</summary>

```python
{'active_tri': 0.125, 'pos_dist': 82.5, 'neg_dist': 100.5}  # batch 1
{'active_tri': 0.0, 'pos_dist': 36.3, 'neg_dist': 56.9}     # batch 2

{'cmc': {1: 0.75}, 'precision': {5: 0.75}, 'map': {3: 0.8}}

```

<img src="https://i.ibb.co/MVxBf80/retrieval-img.png" height="200px">

</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fr4HhDOqmjx1hCFS30G3MlYjeqBW5vDg?usp=sharing)

</td>

<td>

<details style="padding-bottom: 10px">
<summary>Output</summary>

```python
{'active_tri': 0.0, 'pos_dist': 8.5, 'neg_dist': 11.0}  # batch 1
{'active_tri': 0.25, 'pos_dist': 8.9, 'neg_dist': 9.8}  # batch 2

{'cmc': {1: 0.8}, 'precision': {5: 0.7}, 'map': {3: 0.9}}

```

<img src="https://i.ibb.co/HqfXdYd/text-retrieval.png" height="200px">

</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19o2Ox2VXZoOWOOXIns7mcs0aHJZgJWeO?usp=sharing)

</td>

</tr>

</table>

</div>

<br>

[Extra illustrations, explanations and tips](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction#training)
for the code above.

### Retrieval by trained model

Here is an inference time example (in other words, retrieval on test set).
The code below works for both texts and images.

<details>
<summary><b>See example</b></summary>
<p>

[comment]:usage-retrieval-start
```python
from oml.datasets import ImageQueryGalleryDataset
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.utils import get_mock_images_dataset
from oml.retrieval import RetrievalResults, AdaptiveThresholding

_, df_test = get_mock_images_dataset(global_paths=True)
del df_test["label"]  # we don't need gt labels for doing predictions

extractor = ViTExtractor.from_pretrained("vits16_dino").to("cpu")
transform, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImageQueryGalleryDataset(df_test, transform=transform)
embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)

rr = RetrievalResults.from_embeddings(embeddings, dataset, n_items=5)
rr = AdaptiveThresholding(n_std=3.5).process(rr)
rr.visualize(query_ids=[0, 1], dataset=dataset, show=True)

# you get the ids of retrieved items and the corresponding distances
print(rr)
```
[comment]:usage-retrieval-end

</details>



### Retrieval by trained model: streaming & txt2im

Here is an example where queries and galleries processed separately.
* First, it may be useful for **streaming retrieval**, when a gallery (index) set is huge and fixed, but
  queries are coming in batches.
* Second, queries and galleries have different natures, for examples, **queries are texts, but galleries are images**.


<details>
<summary><b>See example</b></summary>
<p>

[comment]:usage-streaming-retrieval-start
```python
import pandas as pd

from oml.datasets import ImageBaseDataset
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, ConstantThresholding
from oml.utils import get_mock_images_dataset

extractor = ViTExtractor.from_pretrained("vits16_dino").to("cpu")
transform, _ = get_transforms_for_pretrained("vits16_dino")

paths = pd.concat(get_mock_images_dataset(global_paths=True))["path"]
galleries, queries1, queries2 = paths[:20], paths[20:22], paths[22:24]

# gallery is huge and fixed, so we only process it once
dataset_gallery = ImageBaseDataset(galleries, transform=transform)
embeddings_gallery = inference(extractor, dataset_gallery, batch_size=4, num_workers=0)

# queries come "online" in stream
for queries in [queries1, queries2]:
    dataset_query = ImageBaseDataset(queries, transform=transform)
    embeddings_query = inference(extractor, dataset_query, batch_size=4, num_workers=0)

    # for the operation below we are going to provide integrations with vector search DB like QDrant or Faiss
    rr = RetrievalResults.from_embeddings_qg(
        embeddings_query=embeddings_query, embeddings_gallery=embeddings_gallery,
        dataset_query=dataset_query, dataset_gallery=dataset_gallery
    )
    rr = ConstantThresholding(th=80).process(rr)
    rr.visualize_qg([0, 1], dataset_query=dataset_query, dataset_gallery=dataset_gallery, show=True)
    print(rr)
```
[comment]:usage-streaming-retrieval-end

</details>

## [Pipelines](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines)

Pipelines provide a way to run metric learning experiments via changing only the config file.
All you need is to prepare your dataset in a required format.

See [Pipelines](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/) folder for more details:
* Feature extractor [pipeline](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction)
* Retrieval re-ranking [pipeline](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing)

## [Zoo](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/zoo.html)

### How to use text models?

Here is a lightweight integration with [HuggingFace Transformers](https://github.com/huggingface/transformers) models.
You can replace it with other arbitrary models inherited from [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor).

Note, we don't have our own text models zoo at the moment.

<details style="padding-bottom: 15px">
<summary><b>See example</b></summary>
<p>

```shell
pip install open-metric-learning[nlp]
```

[comment]:zoo-text-start
```python
from transformers import AutoModel, AutoTokenizer

from oml.models import HFWrapper

model = AutoModel.from_pretrained('bert-base-uncased').eval()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
extractor = HFWrapper(model=model, feat_dim=768)

inp = tokenizer(text="Hello world", return_tensors="pt", add_special_tokens=True)
embeddings = extractor(inp)
```
[comment]:zoo-text-end

</p>
</details>

### How to use image models?

You can use an image model from our Zoo or
use other arbitrary models after you inherited it from [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor).

<details style="padding-bottom: 15px">
<summary><b>See example</b></summary>
<p>

[comment]:zoo-image-start
```python
from oml.const import CKPT_SAVE_ROOT as CKPT_DIR, MOCK_DATASET_PATH as DATA_DIR
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained

model = ViTExtractor.from_pretrained("vits16_dino").eval()
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
[comment]:zoo-image-end

</p>
</details>

### Image models zoo

Models, trained by us.
The metrics below are for **224 x 224** images:

|                      model                      | cmc1  |         dataset          |                                              weights                                              |                                                    experiment                                                     |
|:-----------------------------------------------:|:-----:|:------------------------:|:-------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| `ViTExtractor.from_pretrained("vits16_inshop")` | 0.921 |    DeepFashion Inshop    |    [link](https://drive.google.com/file/d/1niX-TC8cj6j369t7iU2baHQSVN3MVJbW/view?usp=sharing)     | [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_inshop) |
|  `ViTExtractor.from_pretrained("vits16_sop")`   | 0.866 | Stanford Online Products |   [link](https://drive.google.com/file/d/1zuGRHvF2KHd59aw7i7367OH_tQNOGz7A/view?usp=sharing)      |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_sop)   |
| `ViTExtractor.from_pretrained("vits16_cars")`   | 0.907 |         CARS 196         |   [link](https://drive.google.com/drive/folders/17a4_fg94dox2sfkXmw-KCtiLBlx-ut-1?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_cars)  |
|  `ViTExtractor.from_pretrained("vits16_cub")`   | 0.837 |       CUB 200 2011       |   [link](https://drive.google.com/drive/folders/1TPCN-eZFLqoq4JBgnIfliJoEK48x9ozb?usp=sharing)    |  [link](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction/extractor_cub)   |

Models, trained by other researchers.
Note, that some metrics on particular benchmarks are so high because they were part of the training dataset (for example `unicom`).
The metrics below are for 224 x 224 images:

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
|       `ViTExtractor.from_pretrained("vits14_dinov2")`        |          0.566           |       0.334        |    0.797     |  0.503   |
|     `ViTExtractor.from_pretrained("vits14_reg_dinov2")`      |          0.566           |       0.332        |    0.795     |  0.740   |
|       `ViTExtractor.from_pretrained("vitb14_dinov2")`        |          0.565           |       0.342        |    0.842     |  0.644   |
|     `ViTExtractor.from_pretrained("vitb14_reg_dinov2")`      |          0.557           |       0.324        |    0.833     |  0.828   |
|       `ViTExtractor.from_pretrained("vitl14_dinov2")`        |          0.576           |       0.352        |    0.844     |  0.692   |
|     `ViTExtractor.from_pretrained("vitl14_reg_dinov2")`      |          0.571           |       0.340        |    0.840     |  0.871   |
|    `ResnetExtractor.from_pretrained("resnet50_moco_v2")`     |          0.493           |       0.267        |    0.264     |  0.149   |
| `ResnetExtractor.from_pretrained("resnet50_imagenet1k_v1")`  |          0.515           |       0.284        |    0.455     |  0.247   |

*The metrics may be different from the ones reported by papers,
because the version of train/val split and usage of bounding boxes may differ.*

## [Contributing guide](https://open-metric-learning.readthedocs.io/en/latest/oml/contributing.html)

We welcome new contributors! Please, see our:
* [Contributing guide](https://open-metric-learning.readthedocs.io/en/latest/oml/contributing.html)
* [Kanban board](https://github.com/OML-Team/open-metric-learning/projects/1)

## Acknowledgments

<a href="https://github.com/catalyst-team/catalyst" target="_blank"><img src="https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/pics/catalyst_logo.png" width="100"/></a>

The project was started in 2020 as a module for [Catalyst](https://github.com/catalyst-team/catalyst) library.
I want to thank people who worked with me on that module:
[Julia Shenshina](https://github.com/julia-shenshina),
[Nikita Balagansky](https://github.com/elephantmipt),
[Sergey Kolesnikov](https://github.com/Scitator)
and others.

I would like to thank people who continue working on this pipeline when it became a separate project:
[Julia Shenshina](https://github.com/julia-shenshina),
[Misha Kindulov](https://github.com/b0nce),
[Aron Dik](https://github.com/dapladoc),
[Aleksei Tarasov](https://github.com/DaloroAT) and
[Verkhovtsev Leonid](https://github.com/leoromanovich).

<a href="https://www.newyorker.de/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/New_Yorker.svg/1280px-New_Yorker.svg.png" width="100"/></a>

I also want to thank NewYorker, since the part of functionality was developed (and used) by its computer vision team led by me.
