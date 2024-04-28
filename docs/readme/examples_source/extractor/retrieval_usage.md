<details>
<summary>Using a trained model for retrieval</summary>
<p>

[comment]:usage-retrieval-start
```python
from oml.datasets import ImageQueryGalleryDataset
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.retrieval.retrieval_results import RetrievalResults


_, df_test = download_mock_dataset(global_paths=True)
del df_test["label"]  # we don't need gt labels for doing predictions

extractor = ViTExtractor.from_pretrained("vits16_dino")
transform, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImageQueryGalleryDataset(df_test, transform=transform)
embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)

retrieval_results = RetrievalResults.compute_from_embeddings(embeddings, dataset, n_items_to_retrieve=5)

retrieval_results.visualize(query_ids=[0, 1], dataset=dataset).show()

print(retrieval_results)  # you get the ids of retrieved items and the corresponding distances

```
[comment]:usage-retrieval-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S2nK6KaReDm-RjjdojdId6CakhhSyvfA?usp=share_link)
