# Pipelines: features extraction

* [What are pipelines?](https://open-metric-learning.readthedocs.io/en/latest/oml/pipelines_general.html)
* Introduction to metric learning:
[English](https://medium.com/@AlekseiShabanov/practical-metric-learning-b0410cda2201) |
[Russian](https://habr.com/ru/company/ods/blog/695380/) |
[Chinese](https://blog.csdn.net/fermion0217/article/details/127932087)


These particular pipelines allow you to train, validate and inference models that represent images as feature vectors.
In this section we explain how the following pipelines work under the hood:
* [extractor_training_pipeline](https://open-metric-learning.readthedocs.io/en/latest/contents/lightning.html#extractor-training-pipeline) including training + validation
* [extractor_validation_pipeline](https://open-metric-learning.readthedocs.io/en/latest/contents/lightning.html#extractor-validation-pipeline) including validation only
* [extractor_prediction_pipeline](https://open-metric-learning.readthedocs.io/en/latest/contents/lightning.html#extractor-prediction-pipeline) including saving extracted features

Pipelines also have the corresponding [analogues](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html) in plain Python.

## Training

It is expected that the dataset will be in the desired
[format](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).
You can see a tiny
[figures](https://drive.google.com/drive/folders/1plPnwyIkzg51-mLUXWTjREHgc1kgGrF4?usp=share_link)
dataset as an example.

To get used to terminology you can check the
[Glossary](https://github.com/OML-Team/open-metric-learning#faq)
(naming convention).


This pipeline support two types of losses:
* Contrastive ones, like [TripletLoss](https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html#tripletlosswithminer).
  They require special
  [Miner](https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html)
  and
  [Batches Sampler](https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html).
  Miner produces triplets exploiting different strategies like
  [hard mining](https://open-metric-learning.readthedocs.io/en/latest/contents/miners.html#hardtripletsminer),
  in its turn Sampler guarantees that batch contains enough
  different labels to form at least one triplet so miner can do its job.

* Classification ones, like [ArcFace](https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html#arcfaceloss).
  They have no mining step by design and batch sampling strategy is optional for them.
  For these losses we consider the output of the layer before the classification head (which is a part of criterion in our implementation)
  as a feature vector.

Note! Despite the different nature of the losses above, they share the same forward signature: `forward(features, labels)`.
That is why mining is happening inside the forward pass, see
[TripletLossWithMiner](https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html#tripletlosswithminer).


<div align="center">
<img src="https://i.ibb.co/FYNkbbg/extractor-train.png">
<div align="left">


## Validation

Validation part consists of the following steps:
1. Accumulating all the embeddings in [EmbeddingMetrics](https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html#embeddingmetrics).
2. Calculating distances between queries and galleries.
3. [Optional] Applying some specific retrieval postprocessing [techniques](https://open-metric-learning.readthedocs.io/en/latest/contents/postprocessing.html) like re-ranking.
4. Calculating retrieval metrics like
   [CMC@k](https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html#calc-cmc),
   [Precision@k](https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html#calc-precision),
   [MeanAveragePrecision@k](https://open-metric-learning.readthedocs.io/en/latest/contents/metrics.html#calc-map)
   or others.

<div align="center">
<img src="https://i.ibb.co/kcVb7YH/extractor-validation.png">
<div align="left">


## Prediction / Inference

Prediction pipeline runs inference of a trained model and saves extracted features to the disk.
Note, to speed up inference you can easily turn on multi GPU setup in the corresponding config file.

## Customization

Pipelines are built around blocks like model, criterion, optimizer and so on.
Some of them can be replaced by existing entities from OML or by your custom implementations, see the customisation
[instruction](https://open-metric-learning.readthedocs.io/en/latest/oml/pipelines_general.html#how-to-use-my-own-implementation-of-loss-model-etc).

In feature extraction pipelines you can customize:

|  Block in config   |       Registry*       |                                       Example configs                                        |                                                                                                                                Requirements on custom implementation                                                                                                                                 |
|:------------------:|:---------------------:|:--------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `transforms_train` | `TRANSFORMS_REGISTRY` | [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/transforms) |                                                                                             Callable, see [available](https://github.com/OML-Team/open-metric-learning/tree/main/oml/transforms/images).                                                                                             |
|  `transforms_val`  | `TRANSFORMS_REGISTRY` | [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/transforms) |                                                                                             Callable, see [available](https://github.com/OML-Team/open-metric-learning/tree/main/oml/transforms/images).                                                                                             |
|    `extractor`     | `EXTRACTORS_REGISTRY` |   [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/model)    |                                         A successor of [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor), see [available](https://open-metric-learning.readthedocs.io/en/latest/contents/models.html).                                         |
|     `sampler`      |  `SAMPLERS_REGISTRY`  |  [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/sampler)   | For losses with mining see [this](https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html). For classification losses: no restrictions, but set `null` for [RandomSampler](https://pytorch.org/docs/stable/data.html?highlight=random+sampler#torch.utils.data.RandomSampler).  |
|    `criterion`     |   `LOSSES_REGISTRY`   | [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/criterion)  | The signature is required: `forward(features, labels)`. For contrastive losses: mining is implemented inside the forward pass. For classification losses: a classification head is a part of criterion. See [available](https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html). |
|    `optimizer`     | `OPTIMIZERS_REGISTRY` | [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/optimizer)  |                                                                                                                                     A regular PyTorch optimizer.                                                                                                                                     |
|    `scheduling`    | `SCHEDULERS_REGISTRY` | [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/scheduler)  |                                                         A regular PyTorch lr scheduler, structured in Lightning [format](https://github.com/OML-Team/open-metric-learning/blob/main/tests/test_runs/test_pipelines/configs/train.yaml#L51).                                                          |
|      `logger`      |  `LOGGERS_REGISTRY`   |   [configs](https://github.com/OML-Team/open-metric-learning/tree/main/oml/configs/logger)   |                                                                                          Child of [IPipelineLogger](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#ipipelinelogger)                                                                                  |


*Use: `from oml.registry import X_REGISTRY`.

## Tips

We left plenty of comments in the [training config](https://github.com/OML-Team/open-metric-learning/blob/main/pipelines/features_extraction/extractor_cars/train_cars.yaml)
for the CARS dataset, so you can start checking it out.


* If you don't know what parameters to pick for
  [BalanceSampler](https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html#balancesampler),
  simply set `n_labels` equal to the median size of your classes, and set `n_instances` as big as your GPU allows for the given `n_labels`.
* Tips for [TripletLossWithMiner](https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html#tripletlosswithminer):
  * The margin value of `0.2` may be a good choice if your extractor produces normalised features.
  * Triplet loss may struggle with a mode collapse: the situation when your loss goes down,
    but then fluctuates on a plateau on the level of margin value, which means that positive and negative distances both equal to zero.
    In this case, you can try to use the [soft version](https://arxiv.org/abs/1703.07737) of triplet loss instead (just set `margin: null`).
    You can also switch between mining strategies (*hard* / *all*).
  * Don't use `margin: null` if you normalise features since it breaks gradients flow (it can be proved mathematically).
* Check out [Logging & Visualization](https://open-metric-learning.readthedocs.io/en/latest/oml/logging.html) to learn more
  about built-in possibilities such as
  tracking metrics, losses, geometric statistics over embeddings, visual inspection of a model's predictions and so on.
