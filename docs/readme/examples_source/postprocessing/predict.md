<details>
<summary>Postprocessor: Predict</summary>
<p>

[comment]:postprocessor-pred-start
```python

from oml.datasets import ImageQueryGalleryDataset
from oml.inference import inference
from oml.models import ConcatSiamese, ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.retrieval import RetrievalResults

_, df_test = download_mock_dataset(global_paths=True)
del df_test["label"]  # we don't need gt labels for doing predictions

extractor = ViTExtractor.from_pretrained("vits16_dino")
transforms, _ = get_transforms_for_pretrained("vits16_dino")

dataset = ImageQueryGalleryDataset(df_test, transform=transforms)

# 1. Let's get top 5 galleries closest to every query...
embeddings = inference(extractor, dataset, batch_size=4, num_workers=0)
rr = RetrievalResults.compute_from_embeddings(embeddings, dataset, n_items_to_retrieve=5)

# 2. ... and let's re-rank first 3 of them
siamese = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])  # Note! Replace it with your trained postprocessor
postprocessor = PairwiseReranker(top_n=3, pairwise_model=siamese, batch_size=4, num_workers=0)
rr_upd = postprocessor.process(rr, dataset=dataset)

# You may see the first 3 positions have changed, but the rest remain the same:
print(rr, "\n", rr_upd)
```
[comment]:postprocessor-pred-end
</p>
</details>
