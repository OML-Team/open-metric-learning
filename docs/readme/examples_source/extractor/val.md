<details>
<summary>Validation</summary>
<p>

[comment]:vanilla-validation-start
```python
import torch
from tqdm import tqdm

from oml.datasets import ImagesDatasetQueryGallery
from oml.metrics.embeddings import EmbeddingMetrics
from oml.models import ViTExtractor
from oml.utils.download_mock_dataset import download_mock_dataset

dataset_root = "mock_dataset/"
_, df_val = download_mock_dataset(dataset_root)

extractor = ViTExtractor("vits16_dino", arch="vits16", normalise_features=False).eval()

val_dataset = ImagesDatasetQueryGallery(df_val, dataset_root=dataset_root)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
calculator = EmbeddingMetrics(dataset=val_dataset)
calculator.setup(num_samples=len(val_dataset))

with torch.no_grad():
    for batch in tqdm(val_loader):
        calculator.update_data(embeddings=extractor(batch["input_tensors"]))

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
