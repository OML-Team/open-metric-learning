[comment]:postprocessor-start
```python

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from oml.datasets import ImageLabeledDataset, ImageQueryGalleryLabeledDataset
from oml.inference import inference_cached
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import PairsMiner
from oml.models import ConcatSiamese, ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset
from oml.transforms.images.torchvision import get_augs_torch
from oml.retrieval import RetrievalResults, PairwiseReranker

# In these example we will train a pairwise model as a re-ranker for ViT
extractor = ViTExtractor.from_pretrained("vits16_dino").to("cpu")
transforms, _ = get_transforms_for_pretrained("vits16_dino")
df_train, df_val = get_mock_images_dataset(global_paths=True)

# STEP 0: SAVE VIT EMBEDDINGS
# - training ones are needed for hard negative sampling when training pairwise model
# - validation ones are needed to construct the original prediction (which we will re-rank)
embeddings_train = inference_cached(extractor, ImageLabeledDataset(df_train, transform=transforms), batch_size=4, num_workers=0)
embeddings_valid = inference_cached(extractor, ImageLabeledDataset(df_val, transform=transforms), batch_size=4, num_workers=0)

# STEP 1: TRAIN PAIRWISE MODEL
train_dataset = ImageLabeledDataset(df_train, transform=get_augs_torch(224), extra_data={"embeddings": embeddings_train})
pairwise_model = ConcatSiamese(extractor=extractor, mlp_hidden_dims=[100])
optimizer = torch.optim.Adam(pairwise_model.parameters(), lr=1e-4)
miner = PairsMiner(hard_mining=True)
criterion = BCEWithLogitsLoss()

train_loader = DataLoader(train_dataset, batch_sampler=BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2))

for batch in train_loader:
    # We sample positive and negative pairs on which the original model struggled most
    ids1, ids2, is_negative_pair = miner.sample(features=batch["embeddings"], labels=batch["labels"])
    probs = pairwise_model(x1=batch["input_tensors"][ids1], x2=batch["input_tensors"][ids2])
    loss = criterion(probs, is_negative_pair.float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# STEP 2: VALIDATE RE-RANKING MODEL (DOES IT IMPROVE METRICS?)
val_dataset = ImageQueryGalleryLabeledDataset(df=df_val, transform=transforms)
rr = RetrievalResults.from_embeddings(embeddings_valid, val_dataset, n_items=5)

reranker = PairwiseReranker(top_n=3, pairwise_model=pairwise_model, num_workers=0, batch_size=4)
rr_upd = reranker.process(rr, dataset=val_dataset)

# STEP 3: comparison
rr.visualize(query_ids=[0, 1], dataset=val_dataset, show=True)
rr_upd.visualize(query_ids=[0, 1], dataset=val_dataset, show=True)

metrics = calc_retrieval_metrics_rr(rr, precision_top_k=(3, 5))
metrics_upd = calc_retrieval_metrics_rr(rr_upd, precision_top_k=(3, 5))
print(f"Before postprocessing:\n{metrics}")
print(f"After postprocessing:\n{metrics_upd}")

```
[comment]:postprocessor-end
