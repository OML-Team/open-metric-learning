<details>
<summary>Postprocessor: Predict</summary>
<p>

[comment]:postprocessor-pred-start
```python
import torch
from torch.utils.data import DataLoader

from oml.const import PATHS_COLUMN
from oml.datasets.base import DatasetQueryGallery
from oml.inference.flat import inference_on_dataframe
from oml.models.meta.siamese import ConcatSiamese
from oml.models.vit.vit import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.retrieval.postprocessors.pairwise import PairwiseImagesPostprocessor
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc_torch import pairwise_dist

dataset_root = "mock_dataset/"
download_mock_dataset(dataset_root)

# 1. Let's use feature extractor to get predictions
extractor = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")

_, emb_val, _, df_val = inference_on_dataframe(dataset_root, "df.csv", extractor, transforms=transforms)

is_query = df_val["is_query"].astype('bool').values
distances = pairwise_dist(x1=emb_val[is_query], x2=emb_val[~is_query])

print("\nOriginal predictions:\n", torch.topk(distances, dim=1, k=3, largest=False)[1])

# 2. Let's initialise a random pairwise postprocessor to perform re-ranking
siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])  # Note! Replace it with your trained postprocessor
postprocessor = PairwiseImagesPostprocessor(top_n=3, pairwise_model=siamese, transforms=transforms)

dataset = DatasetQueryGallery(df_val, extra_data={"embeddings": emb_val}, transform=transforms)
loader = DataLoader(dataset, batch_size=4)

query_paths = df_val[PATHS_COLUMN][is_query].values
gallery_paths = df_val[PATHS_COLUMN][~is_query].values
distances_upd = postprocessor.process(distances=distances, queries=query_paths, galleries=gallery_paths)

print("\nPredictions after postprocessing:\n", torch.topk(distances_upd, dim=1, k=3, largest=False)[1])

```
[comment]:postprocessor-pred-end
</p>
</details>
