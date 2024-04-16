<details>
<summary>Postprocessor: Predict</summary>
<p>

[comment]:postprocessor-pred-start
```python
from oml.const import MOCK_DATASET_PATH
from oml.datasets import ImagesDatasetQueryGallery
from oml.inference import inference
from oml.models import ConcatSiamese, ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.retrieval.prediction import RetrievalPrediction
from oml.utils.download_mock_dataset import download_mock_dataset

_, df_test = download_mock_dataset(MOCK_DATASET_PATH)
df_test["label"] = "fake_label"  # todo 522: we don't use labels in pred, but we need to handle it more elegant later

# 1. Let's use feature extractor to get predictions
extractor = ViTExtractor.from_pretrained("vits16_dino")
transform, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImagesDatasetQueryGallery(df=df_test, dataset_root=MOCK_DATASET_PATH, transform=transform)
embeddings = inference(extractor, dataset, batch_size=8)
prediction = RetrievalPrediction.compute_from_embeddings(embeddings, dataset)

# 2. Let's apply re-ranking model
siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])  # Note! Replace it with your trained postprocessor
postprocessor = PairwiseReranker(top_n=3, pairwise_model=siamese, num_workers=0, batch_size=4)
prediction_upd = postprocessor.process(prediction, dataset=dataset)

# 3. Let's check the difference
print(prediction)
print(prediction_upd)

prediction.visualize(query_ids=[0, 1], dataset=dataset)
prediction_upd.visualize(query_ids=[0, 1], dataset=dataset)

```
[comment]:postprocessor-pred-end
</p>
</details>