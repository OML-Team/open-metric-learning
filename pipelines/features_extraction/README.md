# Pipelines: features extraction

These pipelines allow you to train and validate models that represent images as feature vectors, in other words,
train feature extractors. Basically, there are two pipelines:
* [extractor_training_pipeline](https://open-metric-learning.readthedocs.io/en/latest/contents/lightning.html#extractor-training-pipeline) including training + validation
* [extractor_validation_pipeline](https://open-metric-learning.readthedocs.io/en/latest/contents/lightning.html#extractor-validation-pipeline) including validation only

You can see the [analogues](https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/python_examples.html) in Python.

It is expected that the dataset will be in the desired [format](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).

## Training
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


## Customization

Pipelines are built around blocks like model, criterion, optimizer and so on.
Some of them can be replaced by existing entities from OML or by your custom implementations, see the customisation
[instruction](https://open-metric-learning.readthedocs.io/en/latest/oml/pipelines_general.html#how-to-use-my-own-implementation-of-loss-model-etc).

In feature extraction pipelines you can customize:

|   Name in config   |                                                                                                                                                                    Requirements                                                                                                                                                                    | Where to add my implementation?*  |                                Where to find the existing implementations?                                |
|:------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------:|:---------------------------------------------------------------------------------------------------------:|
| `transforms_train` |                                                                                                              Callable, see [available](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/transforms/images).                                                                                                               |       `TRANSFORMS_REGISTRY`       | [configs](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/configs/transforms)   |
|  `transforms_val`  |                                                                                                              Callable, see [available](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/transforms/images).                                                                                                               |       `TRANSFORMS_REGISTRY`       |  [configs](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/configs/transforms)  |
|    `extractor`     |                                                                A successor of [IExtractor](https://open-metric-learning.readthedocs.io/en/latest/contents/interfaces.html#iextractor), see [available](https://open-metric-learning.readthedocs.io/en/latest/contents/models.html).                                                                |       `EXTRACTORS_REGISTRY`       |    [configs](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/configs/model)     |
|     `sampler`      | For losses with a miner: one guarantees the correct work of a miner, see [available](https://open-metric-learning.readthedocs.io/en/latest/contents/samplers.html). For classification losses: no restrictions, set `null` for [RandomSampler](https://pytorch.org/docs/stable/data.html?highlight=random+sampler#torch.utils.data.RandomSampler). |        `SAMPLERS_REGISTRY`        |   [configs](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/configs/sampler)    |
|    `criterion`     |                   The following signature is required: `forward(features, labels)`. For contrastive losses: mining is implemented inside the forward pass. For classification losses: a classification head is a part of criterion. See [available](https://open-metric-learning.readthedocs.io/en/latest/contents/losses.html).                   |         `LOSSES_REGISTRY`         |  [configs](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/configs/criterion)   |
|    `optimizer`     |                                                                                                                                                            A normal PyTorch optimizer.                                                                                                                                                             |       `OPTIMIZERS_REGISTRY`       |  [configs](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/configs/optimizer)   |
|    `scheduling`    |                                                                           A normal PyTorch lr scheduler, structured in Lightning [format](https://github.com/OML-Team/open-metric-learning/blob/pipeline_readme/tests/test_runs/test_pipelines/configs/train.yaml#L51).                                                                            |       `SCHEDULERS_REGISTRY`       |  [configs](https://github.com/OML-Team/open-metric-learning/tree/pipeline_readme/oml/configs/scheduler)   |
* The exact registry has to be imported via: `from oml.registry import X_REGISTRY`.

## Tips

We left plenty of comments in the [training config](https://github.com/OML-Team/open-metric-learning/blob/pipeline_readme/pipelines/features_extraction/extractor_cars/train_cars.yaml)
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
* Set `log_images: True` to see the images where the model's performance was worst.
* You can analyse your trained extractor in [visualization.ipynb](https://github.com/OML-Team/open-metric-learning/blob/pipeline_readme/pipelines/features_extraction/visualization.ipynb).
