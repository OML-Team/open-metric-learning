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
<a href="https://edgify.ai/" target="_blank"><img src="https://edgify.ai/wp-content/themes/edgifyai/dist/assets/logo.svg" width="100" height="30"/></a>ㅤㅤ
<a href="https://inspector-cloud.ru/" target="_blank"><img src="https://thumb.tildacdn.com/tild6533-6433-4137-a266-613963373637/-/resize/540x/-/format/webp/photo.png" width="100" height="30"/></a>


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


### OML 3.0 has been released!
<details>
<summary>Details</summary>
<p>

The update focuses on two components:

* We introduced the `RetrievalResults` (`RR`) class — a container to store gallery items retrieved for given queries.
`RR` provides a unified way to visualize predictions and compute metrics (if ground truths are known).
It also simplifies post-processing, where an `RR` object is taken as input and another `RR_upd` is produced as output.
With these two objects, you can compare them visually or compute and compare metrics.
Moreover, you can easily create a chain of such post-processors.
  * `RR` optimizes memory consumption using batching: in other words, it doesn't store full matrix of query-gallery distances.
    (It doesn't make search approximate, it remains exact).

* We made `Model` and `Dataset` the only classes responsible for processing modality-specific logic.
`Model` is responsible for interpreting its input dimensions: `BxCxHxW` for images or `BxLxD` for sequences like texts.
`Dataset` is responsible for preparing an item: it may use `Transforms` for images or `Tokenizer` for texts.
Functions computing metrics, `RetrievalResults`, `PairwiseReranker`, and other classes and functions are unified
to work with any modality.
  * We added `IVisualizableDataset` interface requiring `.visaulize()` method for a single item. If implemented,
   `RetrievalResults` is able to show the layout of search results.

#### Migration from OML 2.* [Python API]:

Note, the easiest way to catch up with changes is to re-read the examples!

* The recommended way of validation is to use `RetrievalResults` and functions like `calc_retrieval_metrics_rr`,
`calc_fnmr_at_fmr_rr`, and others. `EmbeddingMetrics` is kept for use with PyTorch Lightning and inside Pipelines.
Note, the signatures of `EmbeddingMetrics` methods have been slightly changed.

* Since modality-specific logic is confined to `Dataset`, it doesn't output `PATHS_KEY`, `X1_KEY`, `X2_KEY`, `Y1_KEY`, and `Y2_KEY` anymore.
Keys which are not modality specific like `LABELS_KEY`, `IS_GALLERY`, `IS_QUERY_KEY`, `CATEGORIES_KEY` are still in use.

* `inference_on_images` is now `inference` and works with any modality.

* Slightly changed interfaces of `Datasets.`

* Removed: `IMetricDDP`, `EmbeddingMetricsDDP`, `calc_distance_matrix`, and other internal objects that shouldn't affect you.

#### Migration from OML 2.* [Pipelines]:

* [Feature extraction](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/features_extraction):
No changes, except for adding an optional argument — `mode_for_checkpointing = (min | max)`.

* [Pairwise-postprocessing pipeline](https://github.com/OML-Team/open-metric-learning/tree/main/pipelines/postprocessing/pairwise_postprocessing):
Slightly changed the name and arguments of the `postprocessor` sub config — `pairwise_images` is now `pairwise_reranker`
and doesn't need transforms.

</p>
</details>
