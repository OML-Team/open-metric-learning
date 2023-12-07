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
use_knn = False
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
