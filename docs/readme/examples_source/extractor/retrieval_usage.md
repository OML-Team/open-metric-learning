<details>
<summary>Using a trained model for retrieval</summary>
<p>

[comment]:usage-retrieval-start
```python
from oml.const import MOCK_DATASET_PATH
from oml.inference import inference
from oml.datasets import ImagesDatasetQueryGallery
from oml.models import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.retrieval.prediction import RetrievalPrediction

_, df_test = download_mock_dataset(MOCK_DATASET_PATH)
df_test["label"] = "fake_label"   # todo 522: we don't use labels in pred, but we need to handle it more elegant later

extractor = ViTExtractor.from_pretrained("vits16_dino")
transform, _ = get_transforms_for_pretrained("vits16_dino")

dataset_test = ImagesDatasetQueryGallery(df=df_test, dataset_root=MOCK_DATASET_PATH, transform=transform)
features = inference(extractor, dataset_test, batch_size=8)

prediction_test = RetrievalPrediction.compute_from_embeddings(features, dataset_test)
print(prediction_test)
prediction_test.visualize(query_ids=[0, 1], dataset=dataset_test).show()

```
[comment]:usage-retrieval-end
</p>
</details>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S2nK6KaReDm-RjjdojdId6CakhhSyvfA?usp=share_link)
