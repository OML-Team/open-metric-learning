`Category` is something that hierarchically unites a group of labels.
For example, we have 5 different catalog items of tables with the `label`s like `table1`, `table2`, `table2`
and their `category` is `tables`.

Information about categories can be used in two ways in OML.

**First**, for training.
* Category balanced sampling may help to deal with category imbalance.
* For contrastive losses, limiting the number of categories in batches also helps to mine harder negative
  samples (another table is harder positive example than another sofa).
  Without such samples there is no guarantee that we get batch full of tables.

**Second**, for validation.
* Having categories allows to obtain fine-grained metrics and recognize over- and under- performing subsets of the dataset.

<details>
<summary><b>SEE EXAMPLE</b></summary>

[comment]: categories-start
```python
from pprint import pprint

import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults
from oml.samplers import DistinctCategoryBalanceSampler, CategoryBalanceSampler
from oml.utils import get_mock_images_dataset

model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").train()
transform, _ = get_transforms_for_pretrained("vits16_dino")

df_train, df_val = get_mock_images_dataset(df_name="df_with_category.csv", global_paths=True)
train = d.ImageLabeledDataset(df_train, transform=transform)
val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)

# >>>>> You can use one of category aware samplers
args = {"n_categories": 2, "n_labels": 2, "n_instances": 2, "label2category": train.get_label2category(), "labels": train.get_labels()}
sampler = DistinctCategoryBalanceSampler(epoch_size=5, **args)
# sampler = CategoryBalanceSampler(resample_labels=False, weight_categories=True, **args)  # a bit different sampling


def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pprint(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)

    # >>>> When query categories are known we may get fine-grained metrics
    query_categories = np.array(val.extra_data["category"])[val.get_query_ids()]
    pprint(calc_retrieval_metrics_rr(rr, query_categories=query_categories, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()

```
[comment]: categories-end

</details>
<br>


