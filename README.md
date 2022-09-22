<div align="center">
<img src="https://i.ibb.co/wsmD5r4/photo-2022-06-06-17-40-52.jpg" width="400px">

![example workflow](https://github.com/OML-Team/open-metric-learning/actions/workflows/test-pypi.yaml/badge.svg?)
![example workflow](https://github.com/OML-Team/open-metric-learning/actions/workflows/pre-commit-workflow.yaml/badge.svg)
![example workflow](https://github.com/OML-Team/open-metric-learning/actions/workflows/tests-workflow.yaml/badge.svg?)
[![Pipi version](https://img.shields.io/pypi/v/open-metric-learning.svg)](https://pypi.org/project/open-metric-learning/)
[![PyPI Status](https://pepy.tech/badge/open-metric-learning)](https://pepy.tech/project/open-metric-learning)
[![Documentation Status](https://readthedocs.org/projects/open-metric-learning/badge/?version=latest)](https://open-metric-learning.readthedocs.io/en/latest/?badge=latest)

<div align="left">

OML is a PyTorch-based framework to train and validate the models producing high-quality embeddings.

Specifically, our framework provides modules for supervised training and retrieval-like validation, and a single pipeline for them.
* **Training part** implies using losses, well-established for metric learning, such as the angular losses
 (like ArcFace) or the combinations based losses (like TripletLoss or ContrastiveLoss).
 The latter benefits from effective mining schemas of triplets/pairs, so we pay great attention to it.
 Thus, during the training we:
   1. Use `DataLoader` + `Sampler` to form batches (for example `BalanceSampler`)
   2. [Only for losses based on combinations] Use `Miner` to form effective pairs or triplets, including
   those which utilize a memory bank.
   3. Compute loss.
* **Validation part** consists of several stages:
  1. Accumulating all of the embeddings (`EmbeddingMetrics`).
  2. Calculating distances between them with respect to query/gallery split.
  3. Applying some specific retrieval techniques like query reranking or score normalisation.
  4. Calculating retrieval metrics like CMC@k, Precision@k or MeanAveragePrecision.

## Documentation
Documentation is available via the [link](https://open-metric-learning.readthedocs.io/en/latest/index.html).


## Get started using Config API
Using configs is the best option if your dataset and pipeline are standard enough or if you are not
experienced in Machine Learning or Python. You can find more details in the
[examples](https://github.com/OML-Team/open-metric-learning/blob/main/examples/).





