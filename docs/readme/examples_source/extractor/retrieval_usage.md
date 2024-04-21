<details>
<summary>Using a trained model for retrieval</summary>
<p>

[comment]:usage-retrieval-start
```python
import torch

from oml.datasets import ImageQueryGalleryDataset
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc_torch import pairwise_dist

_, df_test = download_mock_dataset(global_paths=True)
del df_test["label"]  # we don't need gt labels for doing predictions

extractor = ViTExtractor.from_pretrained("vits16_dino")
transform, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImageQueryGalleryDataset(df_test, transform=transform)

embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)
embeddings_query = embeddings[dataset.get_query_ids()]
embeddings_gallery = embeddings[dataset.get_gallery_ids()]

# Now we can explicitly build pairwise matrix of distances or save you RAM via using kNN
use_knn = False
top_k = 3

if use_knn:
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(algorithm="auto", p=2).fit(embeddings_query)
    dists, ii_closest = knn.kneighbors(embeddings_gallery, n_neighbors=top_k, return_distance=True)

else:
    dist_mat = pairwise_dist(x1=embeddings_query, x2=embeddings_gallery, p=2)
    dists, ii_closest = torch.topk(dist_mat, dim=1, k=top_k, largest=False)

print(f"Top {top_k} items closest to queries are:\n {ii_closest}")
```
[comment]:usage-retrieval-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S2nK6KaReDm-RjjdojdId6CakhhSyvfA?usp=share_link)
