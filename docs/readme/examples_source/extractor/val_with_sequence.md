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
import torch
from tqdm import tqdm

from oml.datasets.base import DatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models import ViTExtractor
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
_, df_val = download_mock_dataset(dataset_root, df_name="df_with_sequence.csv")  # <- sequence info is in the file

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).eval()

val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)
calculator = EmbeddingMetrics(dataset=val_dataset)
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        batch["embeddings"] = extractor(batch["input_tensors"])
        calculator.update_data(batch)

metrics = calculator.compute_metrics()

```
[comment]:val-with-sequence-end
</p>
</details>
