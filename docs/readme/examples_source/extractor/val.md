<details>
<summary>Validation</summary>
<p>

[comment]:vanilla-validation-start
```python

import numpy as np

from oml.datasets import ImageQueryGalleryLabeledDataset
from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr
from oml.models import ViTExtractor
from oml.retrieval import RetrievalResults
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.registry.transforms import get_transforms_for_pretrained

extractor = ViTExtractor.from_pretrained("vits16_dino").to("cpu")
transform, _ = get_transforms_for_pretrained("vits16_dino")

_, df_val = download_mock_dataset(global_paths=True, df_name="df_with_category.csv")
dataset = ImageQueryGalleryLabeledDataset(df_val, transform=transform)
embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)

rr = RetrievalResults.compute_from_embeddings(embeddings, dataset, n_items_to_retrieve=5)
rr.visualize(query_ids=[2, 1], dataset=dataset, show=True)

# you can optionally provide categories to have category wise metrics
query_categories = np.array(dataset.extra_data["category"])[dataset.get_query_ids()]
metrics = calc_retrieval_metrics_rr(rr, query_categories, map_top_k=(3, 5), precision_top_k=(5,), cmc_top_k=(3,))
print(rr, "\n", metrics)

```
[comment]:vanilla-validation-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O2o3k8I8jN5hRin3dKnAS3WsgG04tmIT?usp=sharing)
