<details>
<summary>Postprocessor: Predict</summary>
<p>

[comment]:postprocessor-pred-start
```python
import torch

from oml.datasets import ImageQueryGalleryDataset
from oml.inference import inference
from oml.models import ConcatSiamese, ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc_torch import pairwise_dist

_, df_test = download_mock_dataset(global_paths=True)
del df_test["label"]  # we don't need gt labels for doing predictions

extractor = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImageQueryGalleryDataset(df_test, transform=transforms)

# 1. Let's get top 5 galleries closest to every query...
embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)
embeddings_query = embeddings[dataset.get_query_ids()]
embeddings_gallery = embeddings[dataset.get_gallery_ids()]

distances = pairwise_dist(x1=embeddings_query, x2=embeddings_gallery, p=2)
ii_closest = torch.topk(distances, dim=1, k=5, largest=False)[1]

# 2. ... and let's re-rank first 3 of them
siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])  # Note! Replace it with your trained postprocessor
postprocessor = PairwiseReranker(top_n=3, pairwise_model=siamese, batch_size=4, num_workers=0)
distances_upd = postprocessor.process(distances, dataset=dataset)
ii_closest_upd = torch.topk(distances_upd, dim=1, k=5, largest=False)[1]

# You may see the first 3 positions have changed, but the rest remain the same:
print("\Closest galleries:\n", ii_closest)
print("\nClosest galleries updates:\n", ii_closest_upd)
```
[comment]:postprocessor-pred-end
</p>
</details>
