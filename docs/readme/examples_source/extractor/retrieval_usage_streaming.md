Here is an example where queries and galleries processed separately.
* First, it may be useful for **streaming retrieval**, when a gallery (index) set is huge and fixed, but
  queries are coming in batches.
* Second, queries and galleries have different natures, for examples, queries are texts, but galleries are images.


<details>
<summary><b>See example</b></summary>
<p>

[comment]:usage-streaming-retrieval-start
```python
import pandas as pd

from oml.datasets import ImageBaseDataset
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, ConstantThresholding
from oml.utils import get_mock_images_dataset

extractor = ViTExtractor.from_pretrained("vits16_dino").to("cpu")
transform, _ = get_transforms_for_pretrained("vits16_dino")

paths = pd.concat(get_mock_images_dataset(global_paths=True))["path"]
galleries, queries1, queries2 = paths[:20], paths[20:22], paths[22:24]

# the gallery is "huge" and fixed, so we only process it once
dataset_gallery = ImageBaseDataset(galleries, transform=transform)
embeddings_gallery = inference(extractor, dataset_gallery, batch_size=4, num_workers=0)

# queries come "online" in steam
for queries in [queries1, queries2]:
    dataset_query = ImageBaseDataset(queries, transform=transform)
    embeddings_query = inference(extractor, dataset_query, batch_size=4, num_workers=0)

    # for the operation below we are going to provide integrations with vector search DB like QDrant or Faiss
    rr = RetrievalResults.from_embeddings_qg(
        embeddings_query=embeddings_query, embeddings_gallery=embeddings_gallery,
        dataset_query=dataset_query, dataset_gallery=dataset_gallery
    )
    rr = ConstantThresholding(th=80).process(rr)
    rr.visualize_qg([0, 1], dataset_query=dataset_query, dataset_gallery=dataset_gallery, show=True)
    print(rr)
```
[comment]:usage-streaming-retrieval-end

</details>
