

[comment]:algo-pp-start
```python
from oml.datasets import ImageQueryGalleryLabeledDataset
from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding, ConstantThresholding
from oml.utils import get_mock_images_dataset

_, df_test = get_mock_images_dataset(global_paths=True)

extractor = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImageQueryGalleryLabeledDataset(df_test)

embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)

rr = RetrievalResults.from_embeddings(embeddings, dataset, n_items=5)
print(rr, calc_retrieval_metrics_rr(rr), "\n")

rr = ConstantThresholding(th=75).process(rr)
print(rr, calc_retrieval_metrics_rr(rr), "\n")

rr = AdaptiveThresholding(n_std=1.5).process(rr)
print(rr, calc_retrieval_metrics_rr(rr), "\n")
```
[comment]:algo-pp-end

**Plans**

We welcome contributions to this section. Please, check the repo's issues.
We have plans to implement:
* Query Expand
* Multi Query processing
* Score Normalisation
* PCA
* and more
