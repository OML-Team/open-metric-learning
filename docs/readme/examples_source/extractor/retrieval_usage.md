Here is an inference time example (in other words, retrieval on test set).
The code below works for both texts and images.

<details>
<summary><b>See example</b></summary>
<p>

[comment]:usage-retrieval-start
```python
from oml.datasets import ImageQueryGalleryDataset
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.utils import get_mock_images_dataset
from oml.retrieval import RetrievalResults, AdaptiveThresholding

_, df_test = get_mock_images_dataset(global_paths=True)
del df_test["label"]  # we don't need gt labels for doing predictions

extractor = ViTExtractor.from_pretrained("vits16_dino").to("cpu")
transform, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImageQueryGalleryDataset(df_test, transform=transform)
embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)

rr = RetrievalResults.from_embeddings(embeddings, dataset, n_items=5)
rr = AdaptiveThresholding(n_std=3.5).process(rr)
rr.visualize(query_ids=[0, 1], dataset=dataset, show=True)

# you get the ids of retrieved items and the corresponding distances
print(rr)
```
[comment]:usage-retrieval-end

</details>


