The below is mostly related to animal or person re-identification tasks, where observations
are often done in the form of sequences of frames. The problem appears when calculating retrieval metrics,
because the closest retrieved images most likely will be  the neighbor frames from the same sequence as a query.
Thus, we get good values of metrics. but don't really understand what is going on.
So, it's better to ignore photos taken from the same sequence as a given query.

If you take a look at standard Re-id benchmarks as [MARS](https://zheng-lab.cecs.anu.edu.au/Project/project_mars.html)
dataset, you may see that ignoring frames from the same camera is a part of the actual protocol.
Following the same logic, we introduced `sequence` field in our dataset [format](https://open-metric-learning.readthedocs.io/en/latest/oml/data.html).

**If sequence ids are provided, retrieved items having the same sequence id as a given query will be ignored.**

Below is an example of how to label consecutive shoots of the tiger with the same `sequence`:

<img src="https://i.ibb.co/Q6zwdfZ/tigers1.png">

On the figure below we show how provided sequence labels affect metrics calculation:

<img src="https://i.ibb.co/FbHBfzb/tigers2.png">

| metric      | consider sequence?  | value |
|-------------|---------------------|-------|
| CMC@1       | no (top figure)     | 1.0   |
| CMC@1       | yes (bottom figure) | 0.0   |
| Precision@2 | no (top figure)     | 0.5   |
| Precision@2 | yes (bottom figure) | 0.5   |

To use this functionality you only need to provide `sequence` column in your dataframe
(containing **strings** or **integers**) and pass `sequence_key` to `EmbeddingMetrics()`:

<details>
<summary>Validation + handling sequences</summary>
<p>

[comment]:val-with-sequence-start
```python

from oml.inference import inference
from oml.datasets import ImageQueryGalleryLabeledDataset
from oml.models import ViTExtractor
from oml.retrieval import RetrievalResults
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.metrics import calc_retrieval_metrics_rr

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).eval()

_, df_val = download_mock_dataset(global_paths=True, df_name="df_with_sequence.csv")  # <- sequence info is in the file
dataset = ImageQueryGalleryLabeledDataset(df_val)

embeddings = inference(extractor, dataset, batch_size=4)

rr = RetrievalResults.compute_from_embeddings(embeddings, dataset, n_items_to_retrieve=5)
metrics = calc_retrieval_metrics_rr(rr, map_top_k=(3, 5), precision_top_k=(5,), cmc_top_k=(3,))

print(rr, "\n", metrics)
rr.visualize(query_ids=[2, 1], dataset=dataset, show=True)

```
[comment]:val-with-sequence-end
</p>
</details>
