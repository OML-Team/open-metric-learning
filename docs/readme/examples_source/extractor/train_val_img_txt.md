<table style="width: 100%; border-collapse: collapse; border-spacing: 0; margin: 0; padding: 0;">
    <tr>
        <th style="border: 1px solid black; padding: 0;">IMAGES</th>
        <th style="border: 1px solid black; padding: 0;">TEXTS</th>
    </tr>
    <tr>
        <td rowspan="2" style="border: 1px solid black; padding: 0;">

[comment]:train-val-img-start
```python
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
from oml.samplers import BalanceSampler
from oml.utils import get_mock_images_dataset

model = ViTExtractor.from_pretrained("vits16_dino").to("cpu").train()
transform, _ = get_transforms_for_pretrained("vits16_dino")

df_train, df_val = get_mock_images_dataset(global_paths=True)
train = d.ImageLabeledDataset(df_train, transform=transform)
val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)
def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)
    print(calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()

```
[comment]:train-val-img-end

<img src="https://i.ibb.co/MVxBf80/retrieval-img.png" width="400px">

</td>
<td style="border: 1px solid black; padding: 0;">

[comment]:train-val-txt-start
```python
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import HFWrapper
from oml.retrieval import RetrievalResults
from oml.samplers import BalanceSampler
from oml.utils import get_mock_texts_dataset

model = HFWrapper(AutoModel.from_pretrained("bert-base-uncased"), 768).to("cpu").train()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

df_train, df_val = get_mock_texts_dataset()
train = d.TextLabeledDataset(df_train, tokenizer=tokenizer)
val = d.TextQueryGalleryLabeledDataset(df_val, tokenizer=tokenizer)

optimizer = Adam(model.parameters(), lr=1e-4)
criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
sampler = BalanceSampler(train.get_labels(), n_labels=2, n_instances=2)


def training():
    for batch in DataLoader(train, batch_sampler=sampler):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(criterion.last_logs)


def validation():
    embeddings = inference(model, val, batch_size=4, num_workers=0)
    rr = RetrievalResults.from_embeddings(embeddings, val, n_items=3)
    rr.visualize(query_ids=[2, 1], dataset=val, show=True)
    print(calc_retrieval_metrics_rr(rr, map_top_k=(3,), cmc_top_k=(1,)))


training()
validation()

```
[comment]:train-val-txt-end

<img src="https://i.ibb.co/Vq81ZV1/retrieval-txt.png" width="400px">

</td>
</tr>
</table>

[![Open IMAGES example in Colab](https://colab.research.google.com/drive/1Fr4HhDOqmjx1hCFS30G3MlYjeqBW5vDg?usp=sharing)
[![Open TEXTS example In Colab](https://colab.research.google.com/drive/19o2Ox2VXZoOWOOXIns7mcs0aHJZgJWeO?usp=sharing)
